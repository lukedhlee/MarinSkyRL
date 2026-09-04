"""Policy loss masks, reductions, and global denominators.

Adapted from VERL's ``trainer/ppo/core_algos.py`` (ByteDance and Hugging Face),
licensed under Apache 2.0.
"""

from __future__ import annotations

from typing import Literal, Optional, Sequence, TypeAlias

import torch

from skyrl_train.utils.policy_math import masked_mean, right_pad_to_match


SPAN_THINK_TAG: int = 1  # mirrors span_tagger.SPAN_THINK (kept local to avoid a torch-free import cycle)
TOKEN_MEAN_LOSS_REDUCTION = "token_mean"
SEQUENCE_MEAN_LOSS_REDUCTION = "sequence_mean"
SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION = "seq_mean_token_sum_norm"
GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION = "seq_mean_token_sum_norm_global"
PROMPT_MEAN_LOSS_REDUCTION = "prompt_mean"
LossReduction: TypeAlias = Literal[
    "token_mean",
    "sequence_mean",
    "seq_mean_token_sum_norm",
    "seq_mean_token_sum_norm_global",
    "prompt_mean",
]
SUPPORTED_LOSS_REDUCTIONS: tuple[LossReduction, ...] = (
    TOKEN_MEAN_LOSS_REDUCTION,
    SEQUENCE_MEAN_LOSS_REDUCTION,
    SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION,
    GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION,
    PROMPT_MEAN_LOSS_REDUCTION,
)
# Reductions whose per-micro-batch term is already a normalized SUM (the normalizer was applied
# before the worker saw the batch), so summing the micro-batch terms of one optimizer step yields
# the intended objective and callers must NOT divide the policy term by the accumulation steps.
PRESCALED_SUM_LOSS_REDUCTIONS: tuple[LossReduction, ...] = (
    GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION,
    PROMPT_MEAN_LOSS_REDUCTION,
)


def build_think_weighted_loss_mask(
    loss_mask: Optional[torch.Tensor],
    response_span_tags: Optional[torch.Tensor],
    think_token_weight: float,
) -> Optional[torch.Tensor]:
    """Return a per-token loss-weight mask that down-weights THINK tokens.

    Args:
        loss_mask: the 0/1 (or already-weighted) loss mask, shape (B, A).
        response_span_tags: per-token span tags (SPAN_THINK==1), shape (B, A) or
            None. Tagged 1:1 with the response tokens (== loss_mask layout).
        think_token_weight: weight applied to THINK tokens (1.0 == no-op).

    Returns:
        - `loss_mask` UNCHANGED (same object) when ``think_token_weight == 1.0``
          or ``response_span_tags is None`` or ``loss_mask is None`` — the
          byte-identical default/flag-off path.
        - Otherwise a NEW float tensor equal to ``loss_mask`` everywhere except
          THINK positions, which are multiplied by ``think_token_weight``
          (down-weighted but, for weight > 0, still counted in the support).
    """
    if loss_mask is None or response_span_tags is None or think_token_weight == 1.0:
        return loss_mask

    # Per-token multiplier: think_token_weight on THINK tokens, 1.0 elsewhere.
    # Align tags to the loss_mask response width defensively (right-padded).
    tags = right_pad_to_match(response_span_tags, loss_mask)
    is_think = tags == SPAN_THINK_TAG
    weight = torch.where(
        is_think,
        torch.as_tensor(think_token_weight, dtype=torch.float32, device=loss_mask.device),
        torch.ones((), dtype=torch.float32, device=loss_mask.device),
    )
    return loss_mask.to(torch.float32) * weight


def reduce_loss(
    loss: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_reduction: LossReduction,
    max_seq_len: Optional[int] = None,
    global_denom: Optional[float] = None,
) -> torch.Tensor:
    if loss_reduction == TOKEN_MEAN_LOSS_REDUCTION:
        # sum over *all* valid tokens, divide by total valid-token count
        loss = masked_mean(loss, loss_mask)
    elif loss_reduction == SEQUENCE_MEAN_LOSS_REDUCTION:
        # per-sequence token-mean (dim=-1), then batch-mean
        loss = masked_mean(loss, loss_mask, dim=-1).mean()
    elif loss_reduction == SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION:
        # per-sequence token-sum, normalized by the max sequence length, then batch mean
        # this is the Dr. GRPO loss reduction to avoid length bias by normalizing by a constant
        assert max_seq_len is not None, "max_seq_len must be provided for seq_mean_token_sum_norm loss reduction"
        # NOTE: max_seq_len is computed as cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length by default
        if loss_mask is not None:
            seq_losses = torch.sum(loss * loss_mask, dim=-1) / max_seq_len
        else:
            # If no mask, assume all tokens are valid
            seq_losses = torch.sum(loss, dim=-1) / max_seq_len
        loss = torch.mean(seq_losses)
    elif loss_reduction == GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED_LOSS_REDUCTION:
        # Sum each micro-batch numerator against the driver-computed global denominator.
        # This term is already normalized, so callers must not divide by accumulation steps.
        assert global_denom is not None, "global_denom must be provided for seq_mean_token_sum_norm_global"
        if loss_mask is not None:
            loss = torch.sum(loss * loss_mask) / global_denom
        else:
            # If no mask, assume all tokens are valid
            loss = torch.sum(loss) / global_denom
    elif loss_reduction == PROMPT_MEAN_LOSS_REDUCTION:
        # Prompt-level mean: token-mean within each prompt group, then mean over the prompts of the
        # optimizer-step slice. The 1 / (num_prompts * tokens_in_prompt) weights are folded into the
        # advantages on the driver (`compute_prompt_mean_advantage_scale`, where the per-row prompt
        # identity still exists), so here the reduction is a plain masked SUM. Summing the micro-batch
        # terms of one optimizer step reconstructs the prompt mean; callers must not divide by
        # accumulation steps (see `PRESCALED_SUM_LOSS_REDUCTIONS`).
        if loss_mask is not None:
            loss = torch.sum(loss * loss_mask)
        else:
            # If no mask, assume all tokens are valid
            loss = torch.sum(loss)
    else:
        raise ValueError(f"Invalid loss reduction type: {loss_reduction}")
    return loss


def count_nonzero_advantage_seqs(advantages: torch.Tensor) -> float:
    """Number of sequences (rows) carrying at least one non-zero-advantage token.

    Zero-advantage sequences (excluded / k<2 / zero-variance RLOO groups) contribute
    no gradient, so they must not inflate the global loss denominator Z.

    BIT-IDENTICAL under any disjoint row partition: each row's ``abs().sum(dim=-1) > 0``
    is an EXACT all-zero test (a sum of non-negative magnitudes is 0 iff every element
    is exactly 0 -- no float cancellation possible), independent of device, dtype, and
    how the rows are chunked across data-parallel ranks. Hence summing this count over
    disjoint per-rank shards equals computing it once over the full concatenated batch.
    """
    return float((advantages.abs().sum(dim=-1) > 0).sum().item())


def compute_global_loss_denom(advantages: torch.Tensor, max_seq_len: int, ranks_per_dp_group: int) -> float:
    """Compute the collective-free global denominator for sequence-normalized loss.

    Under ``MeshDispatch`` the full batch is split into ``dp_size`` disjoint row-chunks;
    every rank in a data-parallel group receives the same chunk, so the full-group sum
    equals ``ranks_per_dp_group * (nonzero-adv count over the full batch)`` -- because
    the per-chunk counts summed over the dp groups equal the full-batch count
    (:func:`count_nonzero_advantage_seqs` is partition-invariant).

    ``ranks_per_dp_group = world_size // dp_size``. The ``max(., 1.0)`` clamp matches the
    reduction contract so an all-zero-advantage batch still yields a valid denominator.
    """
    global_num_seqs = ranks_per_dp_group * count_nonzero_advantage_seqs(advantages)
    return max(global_num_seqs, 1.0) * max_seq_len


def compute_prompt_mean_advantage_scale(
    loss_mask: torch.Tensor,
    uids: Sequence[str],
    *,
    dp_size: int,
    mini_batch_size_per_rank: int,
    pad_size: int = 0,
) -> torch.Tensor:
    """Per-row advantage scale that turns a masked SUM into the ``prompt_mean`` reduction.

    Mirrors upstream SkyRL's ``prompt_mean`` (``apply_loss_reduction_to_advantages_minibatch``):
    token ``[i, t]`` of prompt ``p`` is weighted ``1 / (num_prompts * tokens_in_prompt_p)``, so the
    sum of the per-token policy loss equals ``mean_p(token-mean within prompt p)``.

    Upstream normalizes over the prompts of each contiguous mini-batch. This fork dispatches the
    training batch as ``dp_size`` contiguous row-chunks of ``len(uids) // dp_size`` rows (one per
    data-parallel rank, ``MeshDispatch.dispatch``), and every rank steps its optimizer after each
    consecutive window of ``mini_batch_size_per_rank`` rows of its chunk (``TrainingBatchIterator``
    + ``accumulation_steps`` in ``PolicyWorkerBase.ppo_train`` / ``MegatronPolicyWorker.ppo_train``).
    So the normalization unit here is the RANK-LOCAL optimizer-step slice: ``num_prompts`` counts the
    distinct prompts of that slice, and the data-parallel gradient average across ranks then yields
    the prompt mean over the whole optimizer step (exact when every rank holds the same number of
    prompts per step, which the completeness check below implies for fixed-size groups).

    Args:
        loss_mask: ``(batch, response_len)`` mask the worker will reduce with (already think-weighted
            if ``think_token_weight != 1``), so ``tokens_in_prompt_p`` matches the worker's mask sum.
        uids: per-row prompt id, same order as the rows of ``loss_mask``. Rows of one prompt must be
            contiguous and must not straddle a chunk or optimizer-step boundary; violations raise.
        dp_size: number of data-parallel row-chunks the batch is split into.
        mini_batch_size_per_rank: rows consumed per optimizer step on one rank
            (``policy_mini_batch_size * n_samples_per_prompt // dp_size``).
        pad_size: number of trailing padding rows (``pad_batch``); they carry no prompt, get scale
            0, and never count as a prompt.

    Returns:
        ``(batch, 1)`` float tensor; ``advantages * scale`` is what the worker should sum.
    """
    batch_size = int(loss_mask.shape[0])
    if len(uids) != batch_size:
        raise ValueError(f"prompt_mean: got {len(uids)} uids for a batch of {batch_size} rows")
    if dp_size < 1 or batch_size % dp_size != 0:
        raise ValueError(f"prompt_mean: batch of {batch_size} rows is not divisible by dp_size={dp_size}")
    if mini_batch_size_per_rank < 1:
        raise ValueError(f"prompt_mean: mini_batch_size_per_rank must be positive, got {mini_batch_size_per_rank}")
    if pad_size < 0 or pad_size > batch_size:
        raise ValueError(f"prompt_mean: invalid pad_size={pad_size} for a batch of {batch_size} rows")
    num_real_rows = batch_size - pad_size
    chunk_size = batch_size // dp_size

    # Contiguous runs of equal uid; a uid re-appearing later is a layout bug, never mis-weight it.
    runs: list[tuple[str, int, int]] = []  # (uid, start, end)
    seen: set[str] = set()
    start = 0
    for row in range(1, num_real_rows + 1):
        if row == num_real_rows or uids[row] != uids[start]:
            uid = uids[start]
            if uid in seen:
                raise ValueError(
                    f"prompt_mean: uid {uid!r} appears in non-contiguous positions (row {start}); "
                    "prompt groups must be contiguous in the training batch"
                )
            seen.add(uid)
            runs.append((uid, start, row))
            start = row

    def slice_index(row: int) -> tuple[int, int]:
        chunk = row // chunk_size
        return chunk, (row - chunk * chunk_size) // mini_batch_size_per_rank

    # Every prompt group must lie inside one (rank, optimizer-step) slice.
    prompts_per_slice: dict[tuple[int, int], int] = {}
    slice_of_run: list[tuple[int, int]] = []
    for uid, run_start, run_end in runs:
        first = slice_index(run_start)
        last = slice_index(run_end - 1)
        if first != last:
            raise ValueError(
                f"prompt_mean: prompt {uid!r} (rows {run_start}:{run_end}) straddles a data-parallel chunk or "
                f"optimizer-step boundary (chunk_size={chunk_size}, mini_batch_size_per_rank="
                f"{mini_batch_size_per_rank}); prompt_mean needs every prompt group inside one rank-local "
                "optimizer step. Choose train_batch_size / policy_mini_batch_size divisible by the policy "
                "dp size (and, with padding, a batch already divisible by dp)."
            )
        slice_of_run.append(first)
        prompts_per_slice[first] = prompts_per_slice.get(first, 0) + 1

    mask = loss_mask.to(torch.float32)
    scale = torch.zeros((batch_size, 1), dtype=torch.float32, device=loss_mask.device)
    for (uid, run_start, run_end), slice_key in zip(runs, slice_of_run):
        prompt_tokens = mask[run_start:run_end].sum().clamp(min=1.0)
        scale[run_start:run_end] = 1.0 / (prompts_per_slice[slice_key] * prompt_tokens)
    return scale
