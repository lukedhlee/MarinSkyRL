"""CPU tests for the OpenAI-compatible inference HTTP boundary."""

import asyncio
import json

from fastapi.responses import StreamingResponse

from skyrl_train.inference_engines import inference_engine_client_http_endpoint as endpoint


class _RawRequest:
    def __init__(self, body):
        self.body = body
        self.headers = {"content-type": "application/json"}

    async def json(self):
        return self.body


class _Backend:
    model_name = "test-model"

    def __init__(self):
        self.payload = None

    async def chat_completion(self, payload):
        self.payload = payload
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 123,
            "model": self.model_name,
            "prompt_token_ids": [11, 12],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "checking",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {"name": "shell", "arguments": '{"cmd":"pwd"}'},
                            }
                        ],
                    },
                    "logprobs": {"content": [{"token": "x", "logprob": -0.25}]},
                    "token_ids": [42],
                    "finish_reason": "tool_calls",
                    "stop_reason": None,
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }

    async def completion(self, payload):  # pragma: no cover - protocol completeness
        raise AssertionError("not called")


async def _body(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def test_streaming_chat_is_buffered_without_losing_rollout_fields(monkeypatch):
    backend = _Backend()
    monkeypatch.setattr(endpoint, "_global_inference_engine_client", backend)
    request_body = {
        "model": backend.model_name,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "logprobs": True,
    }

    response = asyncio.run(endpoint.handle_openai_request(_RawRequest(request_body), "/chat/completions"))

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert backend.payload["json"]["stream"] is False
    assert "stream_options" not in backend.payload["json"]
    assert request_body["stream"] is True  # caller-owned input was not mutated

    body = asyncio.run(_body(response))
    payloads = [line.removeprefix("data: ") for line in body.splitlines() if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(payload) for payload in payloads[:-1]]

    content, finish, usage = chunks
    assert content["object"] == "chat.completion.chunk"
    assert content["prompt_token_ids"] == [11, 12]
    assert content["choices"][0]["token_ids"] == [42]
    assert content["choices"][0]["logprobs"]["content"][0]["logprob"] == -0.25
    delta = content["choices"][0]["delta"]
    assert delta["reasoning_content"] == "checking"
    assert delta["tool_calls"][0]["index"] == 0
    assert finish["choices"][0]["finish_reason"] == "tool_calls"
    assert usage["choices"] == []
    assert usage["usage"]["completion_tokens"] == 1


def test_streaming_backend_error_remains_json(monkeypatch):
    backend = _Backend()

    async def fail(payload):
        return {"error": {"message": "bad request", "code": 400}}

    backend.chat_completion = fail
    monkeypatch.setattr(endpoint, "_global_inference_engine_client", backend)
    request = _RawRequest(
        {
            "model": backend.model_name,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
    )

    response = asyncio.run(endpoint.handle_openai_request(request, "/chat/completions"))

    assert response.status_code == 400
    assert json.loads(response.body)["error"]["message"] == "bad request"
