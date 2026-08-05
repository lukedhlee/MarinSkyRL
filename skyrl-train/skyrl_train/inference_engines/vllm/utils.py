from typing import Dict, Any


def pop_openai_kwargs(engine_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize & remove OpenAI-serving-only kwargs from engine_kwargs.
    """
    openai_kwargs: Dict[str, Any] = {}

    enable_auto_tools = engine_kwargs.pop("enable_auto_tools", engine_kwargs.pop("enable_auto_tool_choice", None))
    if enable_auto_tools is not None:
        openai_kwargs["enable_auto_tools"] = bool(enable_auto_tools)

    # A custom parser must be IMPORTED before `tool_parser` is resolved by name:
    # the plugin module registers itself via @ToolParserManager.register_module.
    # This is not an OpenAIServingChat kwarg, so it is consumed here rather than
    # forwarded (vLLM's server does the same thing for --tool-parser-plugin).
    tool_parser_plugin = engine_kwargs.pop("tool_parser_plugin", None)
    if tool_parser_plugin:
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager

        ToolParserManager.import_tool_parser(tool_parser_plugin)

    tool_parser = engine_kwargs.pop("tool_parser", engine_kwargs.pop("tool_call_parser", None))
    if tool_parser is not None:
        openai_kwargs["tool_parser"] = tool_parser

    # SERVER-SIDE chat-template defaults, merged under any request-level
    # `chat_template_kwargs` by vLLM's renderer.
    #
    # This is the only lever that reaches an EXTERNAL agent. Harbor's
    # `extra_body.chat_template_kwargs` is implemented for terminus_2, openhands
    # and mini_swe_agent only -- there is NO OpenCode path, so setting
    # `enable_thinking: false` in the harbor block is silently ignored for
    # OpenCode, which builds its own OpenAI requests. Measured consequence on
    # g1_diverse_tezos_100k_8b: the model spends its turn in <think> and then
    # emits malformed tool JSON (wrong keys, or truncated mid-object), the call
    # does not parse, and the episode ends after one step.
    #
    # Setting it here applies to EVERY request on the endpoint regardless of
    # which agent issued it, and both OpenAIServingRender and OpenAIServingChat
    # accept the kwarg (vLLM 0.22), which is why it can ride `openai_kwargs`.
    default_chat_template_kwargs = engine_kwargs.pop("default_chat_template_kwargs", None)
    if default_chat_template_kwargs is not None:
        openai_kwargs["default_chat_template_kwargs"] = dict(default_chat_template_kwargs)

    # Sampling params for OpenAI-style requests (Harbor terminal-bench rollouts)
    openai_sampling = engine_kwargs.pop("openai_sampling_params", None)
    if openai_sampling is not None:
        openai_kwargs["openai_sampling_params"] = openai_sampling

    return openai_kwargs
