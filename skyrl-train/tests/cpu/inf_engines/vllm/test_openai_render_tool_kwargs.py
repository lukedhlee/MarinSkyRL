import ast
from pathlib import Path


def test_openai_serving_render_receives_normalized_openai_kwargs() -> None:
    """vLLM 0.22 validates auto tool choice in the render layer."""
    source_path = (
        Path(__file__).parents[4]
        / "skyrl_train"
        / "inference_engines"
        / "vllm"
        / "vllm_engine.py"
    )
    tree = ast.parse(source_path.read_text())

    render_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "OpenAIServingRender"
    ]

    assert len(render_calls) == 1
    assert any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == "openai_kwargs"
        for keyword in render_calls[0].keywords
    )
