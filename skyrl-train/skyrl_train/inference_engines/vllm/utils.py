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

    # Sampling params for OpenAI-style requests (Harbor terminal-bench rollouts)
    openai_sampling = engine_kwargs.pop("openai_sampling_params", None)
    if openai_sampling is not None:
        openai_kwargs["openai_sampling_params"] = openai_sampling

    return openai_kwargs
