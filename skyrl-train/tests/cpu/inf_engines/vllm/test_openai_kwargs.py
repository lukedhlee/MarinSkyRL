import pytest

from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs


def test_pop_openai_kwargs():
    """
    Test pop_openai_kwargs with both primary and alias.
    Ensure OpenAI kwargs are popped, non-OpenAI kwargs are kept.
    """
    engine_kwargs = {
        "enable_auto_tools": 1,
        "tool_parser": "json",
        "other": "keep",
    }
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tools": True, "tool_parser": "json"}
    assert engine_kwargs == {"other": "keep"}

    engine_kwargs = {"enable_auto_tool_choice": 0, "tool_call_parser": "proto"}
    openai_kwargs = pop_openai_kwargs(engine_kwargs)

    assert openai_kwargs == {"enable_auto_tools": False, "tool_parser": "proto"}
    assert engine_kwargs == {}


def test_pop_openai_kwargs_chat_template_content_format():
    """The content format is a serving-layer knob: popped from the engine args and validated.

    Unset leaves it out of the serving kwargs entirely (vLLM's ``auto``), which keeps the
    default engine bring-up unchanged.
    """
    engine_kwargs = {"chat_template_content_format": "string", "other": "keep"}
    assert pop_openai_kwargs(engine_kwargs) == {"chat_template_content_format": "string"}
    assert engine_kwargs == {"other": "keep"}

    assert pop_openai_kwargs({"other": "keep"}) == {}

    with pytest.raises(ValueError, match="chat_template_content_format"):
        pop_openai_kwargs({"chat_template_content_format": "parts"})
