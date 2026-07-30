"""Qwen3-Next GatedDeltaNet (GDN) kernel routing for 80B production RL.

Two problems this module solves, both surfaced by Stage-7/Stage-8 (see
notes/skyrl/stage8_scope.md and stage7_scope.md):

1. **fla masking (always, when the fla overlay is mounted).** The
   `flash-linear-attention==0.5.0` wheel/sdist installed in the Stage-8
   overlay is BROKEN — it ships only `fla/layers` + `fla/models` and drops
   `fla.modules` / `fla.ops` / `fla.utils`. transformers' module-level
   `from fla.modules import FusedRMSNormGated` +
   `from fla.ops.gated_delta_rule import chunk_gated_delta_rule` therefore
   HARD-CRASH the Qwen3-Next modeling import whenever
   `is_flash_linear_attention_available()` returns True. So we MASK it False
   before the modeling module is imported. transformers then falls back to its
   own `Qwen3NextRMSNormGated` + `torch_chunk_gated_delta_rule` — the
   autograd-differentiable pure-torch path that the Stage-7 capstone trained on
   (jobs 596157/596282, finite loss + grad). This mask is REQUIRED for any run
   that mounts the fla overlay; without it the import dies.

2. **FlashQLA fused tilelang kernel (opt-in via SKYRL_GDN_FLASHQLA=1).** The
   pure-torch GDN path is correct but slow (Stage-8: 27x slower at S=8192).
   When SKYRL_GDN_FLASHQLA=1, after a model is constructed we rebind every
   supported Qwen3-Next or Qwen3.5/3.6
   `*GatedDeltaNet.chunk_gated_delta_rule` instance attribute to a FlashQLA
   public autograd-enabled kernel. Requires FlashQLA + TileLang (the original
   Stage-8 overlay or an isolated PyPI install). Falls back to the pure-torch
   path unless `SKYRL_GDN_FLASHQLA_REQUIRED=1` makes absence fatal.

Usage (call BEFORE transformers' qwen3_next modeling module is imported, then
again on each constructed model):

    from skyrl_train.models.qwen3_next_gdn import mask_fla, engage_flashqla
    mask_fla()                 # always — keeps the modeling import from crashing
    ... model = AutoModelForCausalLM.from_pretrained(...) ...
    engage_flashqla(model)     # no-op unless SKYRL_GDN_FLASHQLA=1
"""
import os
import logging

logger = logging.getLogger(__name__)

_FLA_MASKED = False


def mask_fla() -> bool:
    """Force transformers' `is_flash_linear_attention_available()` to False.

    Idempotent. Returns the pre-mask availability value. Must run before the
    qwen3_next modeling module is imported (the bad `from fla...` lines run at
    module scope). Safe to call when fla is absent (the lambda is harmless).
    """
    global _FLA_MASKED
    try:
        import transformers.utils.import_utils as _iu
    except Exception:  # pragma: no cover - transformers always present
        return False
    try:
        _iu.is_flash_linear_attention_available.cache_clear()
    except Exception:
        pass
    try:
        was = _iu.is_flash_linear_attention_available()
    except Exception:
        was = False
    _iu.is_flash_linear_attention_available = lambda: False
    _FLA_MASKED = True
    logger.info("[gdn] masked fla-availability False (was=%s)", was)
    return was


def _build_flashqla_chunk():
    """Load FlashQLA's autograd-enabled GDN drop-in.

    FlashQLA 0.1.2 exposes the same high-level signature used by Transformers'
    Qwen3-Next and Qwen3.5/3.6 modules, including internal Q/K L2 normalization.
    Using that public entry point also avoids depending on the lower-level
    forward/backward return tuple, which changed between FlashQLA releases.
    Returns ``None`` if the isolated kernel layer is unavailable.
    """
    try:
        from flash_qla import chunk_gated_delta_rule
    except Exception as e:  # overlay not mounted / import failed
        logger.warning("[gdn] FlashQLA unavailable (%s); staying on pure-torch GDN", e)
        return None
    chunk_gated_delta_rule._flashqla = True
    return chunk_gated_delta_rule


_FLASHQLA_FN = None
_FLASHQLA_GDN_TYPES = {
    "Qwen3NextGatedDeltaNet",
    "Qwen3_5GatedDeltaNet",
    "Qwen3_5MoeGatedDeltaNet",
}


def engage_flashqla(model) -> int:
    """Rebind supported Qwen3-Next/Qwen3.5 GDN chunk kernels to FlashQLA.

    No-op (returns 0) unless env SKYRL_GDN_FLASHQLA is truthy. Builds the
    FlashQLA shim once (cached). Returns the number of GDN modules rebound.
    Safe to call on unrelated models (returns 0).
    """
    global _FLASHQLA_FN
    if os.environ.get("SKYRL_GDN_FLASHQLA", "0") not in ("1", "true", "True"):
        return 0
    if _FLASHQLA_FN is None:
        _FLASHQLA_FN = _build_flashqla_chunk()
    if _FLASHQLA_FN is None:
        if os.environ.get("SKYRL_GDN_FLASHQLA_REQUIRED", "0") in ("1", "true", "True"):
            raise RuntimeError(
                "SKYRL_GDN_FLASHQLA_REQUIRED=1 but FlashQLA could not be imported"
            )
        return 0
    n = 0
    for m in model.modules():
        if type(m).__name__ in _FLASHQLA_GDN_TYPES:
            m.chunk_gated_delta_rule = _FLASHQLA_FN
            n += 1
    if n:
        logger.info("[gdn] engaged FlashQLA fused kernel on %d GatedDeltaNet modules", n)
    elif os.environ.get("SKYRL_GDN_FLASHQLA_REQUIRED", "0") in ("1", "true", "True"):
        raise RuntimeError(
            "SKYRL_GDN_FLASHQLA_REQUIRED=1 but no supported GatedDeltaNet modules were found"
        )
    return n
