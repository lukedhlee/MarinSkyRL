"""
OpenAI-compatible HTTP endpoint using InferenceEngineClient as backend.

This module provides a FastAPI-based HTTP endpoint that exposes OpenAI's chat completion API
while routing requests to our internal InferenceEngineClient system.

Main functions:
- serve(): Start the HTTP endpoint.
- wait_for_server_ready(): Wait for server to be ready.
- shutdown_server(): Shutdown the server.
"""

import asyncio
import json
import logging
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any, Dict, Optional, Protocol

import fastapi
import requests
import uvicorn
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class CompletionBackend(Protocol):
    """What this endpoint needs of the engine client it serves.

    InferenceEngineClient satisfies it. It is a protocol rather than that concrete type
    because the dependency runs the other way: inference_engine_client imports this module
    for the error types and the server entrypoints below.
    """

    model_name: str

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]: ...

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]: ...


# Global state to hold the inference engine client and backend
_global_inference_engine_client: Optional[CompletionBackend] = None
_global_uvicorn_server: Optional[uvicorn.Server] = None


# Adapted from vllm.entrypoints.openai.protocol.ErrorResponse
class ErrorInfo(BaseModel):
    message: str
    type: str
    param: Optional[str] = None
    code: int


class ErrorResponse(BaseModel):
    error: ErrorInfo


def set_global_state(inference_engine_client: CompletionBackend, uvicorn_server: uvicorn.Server):
    """Set the global inference engine client."""
    global _global_inference_engine_client
    global _global_uvicorn_server
    _global_inference_engine_client = inference_engine_client
    _global_uvicorn_server = uvicorn_server


def _validate_openai_request(request_json: Dict[str, Any], endpoint: str) -> Optional[ErrorResponse]:
    """Common validation for /chat/completions and /completions endpoints."""
    assert endpoint in ["/completions", "/chat/completions"]

    if _global_inference_engine_client is None:
        return ErrorResponse(
            error=ErrorInfo(
                message="Inference engine client not initialized",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
    if "model" not in request_json:
        return ErrorResponse(
            error=ErrorInfo(
                message=f"The field `model` is required in your `{endpoint}` request.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if _global_inference_engine_client.model_name != request_json["model"]:
        # NOTE: `served_model_name` config is now supported in generator.engine_init_kwargs.
        # Both vllm_engine.py and InferenceEngineClient use it for Harbor/LiteLLM compatibility.
        # See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        return ErrorResponse(
            error=ErrorInfo(
                message=f"Model name mismatch: loaded model name {_global_inference_engine_client.model_name} != model name in request {request_json['model']}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if endpoint == "/completions" and "n" in request_json and request_json["n"] > 1:
        # TODO(Charlie): this constraint can be removed when we leave DP routing to
        # inference frameworks. Or we could try to resolve it when needed.
        return ErrorResponse(
            error=ErrorInfo(
                message="n is not supported in SkyRL for /completions request yet, please set n to 1.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if endpoint == "/chat/completions" and "messages" not in request_json:
        return ErrorResponse(
            error=ErrorInfo(
                message="The field `messages` is required in your `/chat/completions` request.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    if endpoint == "/chat/completions" and request_json["messages"] == []:
        return ErrorResponse(
            error=ErrorInfo(
                message="The field `messages` in `/chat/completions` cannot be an empty list.",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
    return None


def _stream_chunk_base(response: Dict[str, Any], *, object_name: str) -> Dict[str, Any]:
    """Return response-level fields suitable for an OpenAI SSE chunk."""
    chunk = {key: value for key, value in response.items() if key not in {"choices", "object", "usage"}}
    chunk["object"] = object_name
    return chunk


def _chat_delta(message: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a complete assistant message into one valid streaming delta."""
    delta = dict(message)
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        # The OpenAI streaming schema requires an index on every tool-call delta,
        # while the non-streaming response schema does not include it.
        indexed_tool_calls = []
        for index, tool_call in enumerate(tool_calls):
            if isinstance(tool_call, dict):
                tool_call = dict(tool_call)
                tool_call.setdefault("index", index)
            indexed_tool_calls.append(tool_call)
        delta["tool_calls"] = indexed_tool_calls
    return delta


def _buffered_sse_events(
    response: Dict[str, Any], request_json: Dict[str, Any], endpoint: str
) -> list[Dict[str, Any] | str]:
    """Convert one completed backend response to OpenAI-compatible SSE events.

    SkyRL's Ray actors return complete, picklable response dictionaries. Some
    agent clients (notably OpenCode's AI SDK) require the HTTP transport to use
    streaming. We therefore keep generation non-streaming behind the proxy and
    frame the completed response as a short buffered stream at this boundary.

    Token IDs, logprobs, prompt IDs, and provider-specific fields remain on the
    content event so Harbor's recording proxy can reconstruct exact rollout
    details for TIS.
    """
    assert endpoint in ["/completions", "/chat/completions"]
    is_chat = endpoint == "/chat/completions"
    object_name = "chat.completion.chunk" if is_chat else "text_completion"
    base = _stream_chunk_base(response, object_name=object_name)
    events: list[Dict[str, Any] | str] = []

    for choice in response.get("choices") or []:
        if not isinstance(choice, dict):
            continue

        # Keep generation metadata on the content event, including extensions
        # such as token_ids, provider_specific_fields, routed_experts, and
        # logprobs. Only terminal fields move to the finish event.
        content_choice = {
            key: value
            for key, value in choice.items()
            if key not in {"message", "text", "finish_reason", "stop_reason"}
        }
        content_choice.setdefault("index", choice.get("index", 0))
        content_choice["finish_reason"] = None
        if is_chat:
            message = choice.get("message")
            content_choice["delta"] = _chat_delta(message if isinstance(message, dict) else {})
        else:
            content_choice["text"] = choice.get("text") or ""

        content_event = dict(base)
        content_event["choices"] = [content_choice]
        events.append(content_event)

        finish_choice: Dict[str, Any] = {
            "index": choice.get("index", 0),
            "finish_reason": choice.get("finish_reason"),
        }
        if "stop_reason" in choice:
            finish_choice["stop_reason"] = choice["stop_reason"]
        if is_chat:
            finish_choice["delta"] = {}
        else:
            finish_choice["text"] = ""
        finish_event = dict(base)
        finish_event["choices"] = [finish_choice]
        events.append(finish_event)

    stream_options = request_json.get("stream_options")
    include_usage = isinstance(stream_options, dict) and stream_options.get("include_usage") is True
    if include_usage and response.get("usage") is not None:
        usage_event = dict(base)
        usage_event["choices"] = []
        usage_event["usage"] = response["usage"]
        events.append(usage_event)

    events.append("[DONE]")
    return events


async def _serialize_sse_events(events: list[Dict[str, Any] | str]) -> AsyncIterator[str]:
    for event in events:
        payload = event if isinstance(event, str) else json.dumps(event, separators=(",", ":"))
        yield f"data: {payload}\n\n"


async def handle_openai_request(raw_request: Request, endpoint: str) -> Response:
    """Handle /completions or /chat/completions request."""
    assert endpoint in ["/completions", "/chat/completions"]
    try:
        request_json = await raw_request.json()

        # SkyRL-side validation
        error_response = _validate_openai_request(request_json, endpoint=endpoint)
        if error_response is not None:
            return JSONResponse(content=error_response.model_dump(), status_code=error_response.error.code)

        # Serialize fastapi.Request because it is not pickable, which causes ray methods to fail.
        wants_stream = request_json.get("stream", False) is True
        backend_request_json = dict(request_json)
        if wants_stream:
            # Async generators cannot cross the Ray actor boundary. Generate a
            # complete response there, then expose it as buffered SSE here.
            backend_request_json["stream"] = False
            # vLLM rejects stream_options when stream is false. We still retain
            # the original options above to decide whether the client requested
            # the terminal usage chunk.
            backend_request_json.pop("stream_options", None)
        payload = {
            "json": backend_request_json,
            "headers": dict(raw_request.headers) if hasattr(raw_request, "headers") else {},
        }
        if endpoint == "/chat/completions":
            response = await _global_inference_engine_client.chat_completion(payload)
        else:
            response = await _global_inference_engine_client.completion(payload)

        if "error" in response or response.get("object", "") == "error":
            # former is vllm format, latter is sglang format
            error_code = response["error"]["code"] if "error" in response else response["code"]
            return JSONResponse(content=response, status_code=error_code)
        elif wants_stream:
            events = _buffered_sse_events(response, request_json, endpoint)
            return StreamingResponse(_serialize_sse_events(events), media_type="text/event-stream")
        else:
            return JSONResponse(content=response)

    except json.JSONDecodeError as e:
        # To catch possible raw_request.json() errors
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Invalid JSON error: {str(e)}",
                type=HTTPStatus.BAD_REQUEST.phrase,
                code=HTTPStatus.BAD_REQUEST.value,
            ),
        )
        return JSONResponse(content=error_response.model_dump(), status_code=HTTPStatus.BAD_REQUEST.value)
    except Exception as e:
        # Include full traceback for debugging
        tb = traceback.format_exc()
        logger.error(f"Error when handling {endpoint} request in SkyRL:\n{tb}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Error when handling {endpoint} request in SkyRL: {str(e)}\n\nTraceback:\n{tb}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return JSONResponse(content=error_response.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value)


def shutdown_server(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """Shutdown the server.

    Args:
        host: Server host.
        port: Server port.
        max_wait_seconds: How long to wait until the server stops listening.

    Raises:
        Exception: If the server is still responding after *max_wait_seconds*.
    """
    if _global_uvicorn_server is not None:
        _global_uvicorn_server.should_exit = True

    health_url = f"http://{host}:{port}/health"

    for i in range(max_wait_seconds):
        try:
            # If this succeeds, server is still alive
            requests.get(health_url, timeout=1)
        except requests.exceptions.RequestException:
            # A network error / connection refused means server is down.
            logger.info(f"Server shut down after {i + 1} seconds")
            return
        time.sleep(1)

    raise Exception(f"Server failed to shut down within {max_wait_seconds} seconds")


def wait_for_server_ready(host: str = "127.0.0.1", port: int = 8000, max_wait_seconds: int = 30) -> None:
    """
    Wait for the HTTP endpoint to be ready by polling the health endpoint.

    Args:
        host: Host where the server is running
        port: Port where the server is running
        max_wait_seconds: Maximum time to wait in seconds

    Raises:
        Exception: If server doesn't become ready within max_wait_seconds
    """
    max_retries = max_wait_seconds
    health_url = f"http://{host}:{port}/health"

    for i in range(max_retries):
        try:
            response = requests.get(health_url, timeout=1)
            if response.status_code == 200:
                logger.info(f"Server ready after {i + 1} attempts ({i + 1} seconds)")
                return
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError):
            if i == max_retries - 1:
                raise Exception(f"Server failed to start within {max_wait_seconds} seconds")
            time.sleep(1)  # Wait 1 second between retries


def create_app() -> fastapi.FastAPI:
    """Create the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: fastapi.FastAPI):
        logger.info("Starting inference HTTP endpoint...")
        yield

    app = fastapi.FastAPI(
        title="InferenceEngine OpenAI-Compatible API",
        description="OpenAI-compatible chat completion API using InferenceEngineClient",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/v1/chat/completions")
    async def chat_completion(raw_request: Request):
        """
        Takes in OpenAI's `ChatCompletionRequest` and returns OpenAI's `ChatCompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        Same for SGLang. SkyRL does not perform field validation beyond `model` and `session_id`,
        and otherwise depends on the underlying engines' validation.

        Make sure you add in `session_id` (a string or an integer) to ensure load balancing and
        sticky routing. The same agentic rollout / session should share the same `session_id` so
        they get routed to the same engine for better prefix caching. If unprovided, we will route
        to a random engine which is not performant.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_openai_request(raw_request, endpoint="/chat/completions")

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        """
        Takes in OpenAI's `CompletionRequest` and returns OpenAI's `CompletionResponse`.

        Note that the specific fields inside the request and response depend on the backend you use.
        If `config.generator.backend` is `vllm`, then the request and response will be vLLM's.
        SkyRL only validates the fields `model` and `session_id`, and otherwise offloads
        field validation to the underlying engines.

        Make sure you add in `session_id` to ensure load balancing and sticky routing. Since
        `request["prompt"]` can be `Union[list[int], list[list[int]], str, list[str]]`, i.e.
        {batched, single} x {string, token IDs}, we follow the following logic for request routing:
        - For batched request: `session_id`, if provided, must have the same length as `request["prompt"]`
          so that each `request["prompt"][i]` is routed based on `session_id[i]`.
        - For single request: `session_id`, if provided, must be a single integer or a singleton
          list, where each `session_id` is a string or an integer.

        API reference:
        - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        - https://docs.sglang.ai/basic_usage/openai_api_completions.html
        """
        return await handle_openai_request(raw_request, endpoint="/completions")

    # Health check endpoint
    # All inference engine replicas are initialized before creating `InferenceEngineClient`, and thus
    # we can start receiving requests as soon as the FastAPI server starts
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    # This handler only catches unexpected server-side exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
        error_response = ErrorResponse(
            error=ErrorInfo(
                message=f"Unhandled exception: {str(exc)}",
                type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            ),
        )
        return JSONResponse(content=error_response.model_dump(), status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value)

    return app


def serve(
    inference_engine_client: CompletionBackend,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
):
    """
    Start the HTTP endpoint.

    Args:
        inference_engine_client: The InferenceEngineClient to use as backend
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 8000)
        log_level: Logging level (default: "info")
    """
    # Create app
    app = create_app()

    # Configure logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))

    logger.info(f"Starting server on {host}:{port}")

    # Run server
    config = uvicorn.Config(app, host=host, port=port, log_level=log_level, access_log=True)
    server = uvicorn.Server(config)

    # Expose server for external shutdown control (tests)
    set_global_state(inference_engine_client, server)

    try:
        # Run until shutdown
        asyncio.run(server.serve())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise
