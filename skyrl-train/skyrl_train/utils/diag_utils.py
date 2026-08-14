"""
Training rollout diagnostics.

Pure-python helpers (no torch) computing per-group GRPO diagnostics and dumping
train rollouts to disk. Wired into `RayPPOTrainer.postprocess_generator_output`
behind `trainer.diag_group_metrics` / `trainer.dump_train_rollouts` (both default
false), and the entire call site is wrapped in try/except — these must never be
able to crash or stall training.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union


def _sample_reward(reward: Union[float, List[float]]) -> float:
    """Extract the per-trajectory scalar reward, mirroring `get_metrics_from_generator_output`.

    Rewards in `generator_output["rewards"]` are either response-level scalars
    (List[float]) or token-level lists (List[List[float]]); for token-level rewards
    the last token's reward signifies the trajectory's reward (same convention as
    the pass@n computation in `skyrl_train.generators.utils`). An empty token-level
    list yields 0.0 rather than raising.
    """
    if isinstance(reward, list):
        return float(reward[-1]) if reward else 0.0
    return float(reward)


def _percentile(sorted_vals: List[float], q: float) -> float:
    """Nearest-rank percentile over an ascending-sorted list. 0.0 on empty input."""
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def compute_rollout_diagnostics(
    rewards: List[float],
    uids: List[Any],
    response_lengths: List[int],
    stop_reasons: Optional[List[Any]],
    n_samples_per_prompt: int,
) -> Dict[str, float]:
    """Compute per-group GRPO rollout diagnostics.

    A sample counts as "correct" iff its trajectory reward > 0 (gsm8k-style binary
    reward; matches the > 0.0 success convention used for pass@n in
    `get_metrics_from_generator_output`).

    Groups are samples sharing a uid. The all_wrong/mixed/all_correct fractions and
    `diag/group_reward_std_mean` are over ALL groups regardless of size;
    the `diag/phat_frac_{k}_of_{G}` histogram (fraction of groups with exactly k
    correct, k = 0..G, G = n_samples_per_prompt) only counts FULL groups of exactly
    G samples so partial groups cannot skew the empirical pass-rate distribution.
    `diag/truncated_frac` is the fraction of samples whose stop_reason is "length"
    (the value the generators emit at the generation/input token cap; other observed
    values are "stop" and "abort").

    Rewards entries may be response-level scalars or token-level lists (see
    `_sample_reward`). Returns {} on empty input.
    """
    rewards = [_sample_reward(r) for r in rewards]
    if not rewards or not uids:
        return {}

    groups: Dict[Any, List[float]] = {}
    for uid, reward in zip(uids, rewards):
        groups.setdefault(uid, []).append(reward)

    num_groups = len(groups)
    all_wrong = 0
    mixed = 0
    all_correct = 0
    std_sum = 0.0
    for group_rewards in groups.values():
        num_correct = sum(1 for r in group_rewards if r > 0.0)
        if num_correct == 0:
            all_wrong += 1
        elif num_correct == len(group_rewards):
            all_correct += 1
        else:
            mixed += 1
        mean = sum(group_rewards) / len(group_rewards)
        std_sum += (sum((r - mean) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5

    metrics = {
        "diag/num_groups": float(num_groups),
        "diag/frac_groups_all_wrong": all_wrong / num_groups,
        "diag/frac_groups_mixed": mixed / num_groups,
        "diag/frac_groups_all_correct": all_correct / num_groups,
        "diag/group_reward_std_mean": std_sum / num_groups,
    }

    g = int(n_samples_per_prompt)
    if g >= 1:
        full_groups = [gr for gr in groups.values() if len(gr) == g]
        correct_counts = [sum(1 for r in gr if r > 0.0) for gr in full_groups]
        for k in range(g + 1):
            frac = correct_counts.count(k) / len(full_groups) if full_groups else 0.0
            metrics[f"diag/phat_frac_{k}_of_{g}"] = frac

    sorted_lens = sorted(float(length) for length in response_lengths)
    metrics["diag/response_len_mean"] = sum(sorted_lens) / len(sorted_lens) if sorted_lens else 0.0
    metrics["diag/response_len_p50"] = _percentile(sorted_lens, 0.50)
    metrics["diag/response_len_p90"] = _percentile(sorted_lens, 0.90)
    metrics["diag/response_len_p99"] = _percentile(sorted_lens, 0.99)
    metrics["diag/response_len_max"] = sorted_lens[-1] if sorted_lens else 0.0

    if stop_reasons:
        truncated = sum(1 for s in stop_reasons if s == "length")
        metrics["diag/truncated_frac"] = truncated / len(stop_reasons)
    else:
        metrics["diag/truncated_frac"] = 0.0

    return metrics


def dump_train_rollouts(
    generator_output: Dict[str, Any],
    uids: List[Any],
    tokenizer,
    export_path: str,
    global_step: int,
) -> None:
    """Dump train rollouts to `<export_path>/diag_rollouts/step_{global_step}.jsonl`.

    One JSON line per sample with decoded prompt/response text, the per-trajectory
    scalar reward (see `_sample_reward`), stop_reason, and response token count.
    Written via a temp file + os.replace so a partially written dump is never
    observed. Must be called BEFORE the trainer re-assigns
    `generator_output["rewards"]` to per-token rewards if response-level rewards
    are to be preserved (the extraction handles both shapes regardless).
    """
    out_dir = os.path.join(export_path, "diag_rollouts")
    os.makedirs(out_dir, exist_ok=True)
    final_path = os.path.join(out_dir, f"step_{global_step}.jsonl")
    tmp_path = final_path + ".tmp"

    prompt_token_ids = generator_output["prompt_token_ids"]
    response_ids = generator_output["response_ids"]
    rewards = generator_output["rewards"]
    stop_reasons = generator_output.get("stop_reasons") or [None] * len(response_ids)

    with open(tmp_path, "w", encoding="utf-8") as f:
        for i in range(len(response_ids)):
            record = {
                "step": global_step,
                "uid": uids[i] if i < len(uids) else None,
                "prompt": tokenizer.decode(prompt_token_ids[i], skip_special_tokens=True),
                "response": tokenizer.decode(response_ids[i], skip_special_tokens=True),
                "reward": _sample_reward(rewards[i]),
                "stop_reason": stop_reasons[i] if i < len(stop_reasons) else None,
                "response_len": len(response_ids[i]),
                "tool_calls": None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, final_path)
