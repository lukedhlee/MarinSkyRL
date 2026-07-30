import pytest

from skyrl_train.models import qwen3_next_gdn


class DummyModel:
    def __init__(self, modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


def make_module(type_name: str):
    return type(type_name, (), {"chunk_gated_delta_rule": None})()


def test_flashqla_binds_qwen3_next_and_qwen3_5(monkeypatch) -> None:
    kernel = object()
    monkeypatch.setenv("SKYRL_GDN_FLASHQLA", "1")
    monkeypatch.setattr(qwen3_next_gdn, "_FLASHQLA_FN", kernel)
    supported = [
        make_module("Qwen3NextGatedDeltaNet"),
        make_module("Qwen3_5GatedDeltaNet"),
        make_module("Qwen3_5MoeGatedDeltaNet"),
    ]
    unrelated = make_module("UnrelatedAttention")

    count = qwen3_next_gdn.engage_flashqla(DummyModel([*supported, unrelated]))

    assert count == 3
    assert all(module.chunk_gated_delta_rule is kernel for module in supported)
    assert unrelated.chunk_gated_delta_rule is None


def test_required_flashqla_fails_loudly_when_kernel_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("SKYRL_GDN_FLASHQLA", "1")
    monkeypatch.setenv("SKYRL_GDN_FLASHQLA_REQUIRED", "1")
    monkeypatch.setattr(qwen3_next_gdn, "_FLASHQLA_FN", None)
    monkeypatch.setattr(qwen3_next_gdn, "_build_flashqla_chunk", lambda: None)

    with pytest.raises(RuntimeError, match="could not be imported"):
        qwen3_next_gdn.engage_flashqla(DummyModel([]))
