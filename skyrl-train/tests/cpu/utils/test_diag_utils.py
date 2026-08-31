"""Per-group GRPO diagnostics and train-rollout dumps (`skyrl_train.utils.diag_utils`).

Run with:
uv run --frozen pytest tests/cpu/utils/test_diag_utils.py
"""

import json

import pytest

from skyrl_train.utils import diag_utils


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def test_empty_input_yields_no_metrics():
    assert diag_utils.compute_group_diagnostics([], [], [], [], None, 4) == {}


def test_mismatched_lengths_yield_no_metrics():
    assert diag_utils.compute_group_diagnostics([1.0, 0.0], [True], ["a", "a"], [3, 3], None, 2) == {}


def test_group_learnability_fractions_and_phat_histogram():
    # Three full groups of G=2: all-wrong, mixed, all-correct; plus one partial group.
    uids = ["a", "a", "b", "b", "c", "c", "d"]
    rewards = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    successes = [r > 0.0 for r in rewards]
    lengths = [10, 20, 30, 40, 50, 60, 70]
    stop = ["stop", "length", "stop", "stop", "length", "stop", "stop"]

    m = diag_utils.compute_group_diagnostics(rewards, successes, uids, lengths, stop, n_samples_per_prompt=2)

    assert m["diag/num_groups"] == 4.0
    assert m["diag/frac_groups_all_wrong"] == pytest.approx(1 / 4)
    assert m["diag/frac_groups_mixed"] == pytest.approx(1 / 4)
    assert m["diag/frac_groups_all_correct"] == pytest.approx(2 / 4)  # "d" is a singleton all-correct group
    # Zero-std groups: a (0,0), c (1,1), d (1). Only b carries a GRPO signal.
    assert m["diag/frac_groups_zero_reward_std"] == pytest.approx(3 / 4)
    assert m["diag/group_reward_std_mean"] == pytest.approx(0.5 / 4)
    # Histogram counts only the three full groups; the singleton "d" is excluded.
    assert m["diag/phat_frac_0_of_2"] == pytest.approx(1 / 3)
    assert m["diag/phat_frac_1_of_2"] == pytest.approx(1 / 3)
    assert m["diag/phat_frac_2_of_2"] == pytest.approx(1 / 3)
    assert m["diag/out_tok_p50"] == 40.0
    assert m["diag/out_tok_p99"] == 70.0
    assert m["diag/truncated_fraction"] == pytest.approx(2 / 7)


def test_successes_are_independent_of_optimization_rewards():
    # Reward shaping can give every trajectory a distinct reward while the task outcome is
    # uniform; the learnability split follows `successes`, the std follows `rewards`.
    uids = ["a", "a"]
    rewards = [0.3, 0.7]
    successes = [False, False]
    m = diag_utils.compute_group_diagnostics(rewards, successes, uids, [1, 1], None, 2)
    assert m["diag/frac_groups_all_wrong"] == 1.0
    assert m["diag/frac_groups_zero_reward_std"] == 0.0
    assert m["diag/group_reward_std_mean"] == pytest.approx(0.2)
    assert m["diag/truncated_fraction"] == 0.0


def test_token_level_rewards_are_summed_per_trajectory():
    rewards = [[0.0, 0.0, 1.0], [0.0, 0.5, 0.5]]
    m = diag_utils.compute_group_diagnostics(rewards, [True, True], ["a", "a"], [3, 3], None, 2)
    assert m["diag/frac_groups_all_correct"] == 1.0
    assert m["diag/frac_groups_zero_reward_std"] == 1.0  # both sum to 1.0


def test_dump_train_rollouts_writes_one_record_per_trajectory(tmp_path):
    batch = {
        "prompt_token_ids": [[1, 2], [3]],
        "response_ids": [[4, 5, 6], [7]],
        "rewards": [1.0, [0.0, 0.25]],
        "unshaped_rewards": [1.0, 0.0],
        "stop_reasons": ["stop", "length"],
        "is_last_step": None,
    }
    path = diag_utils.dump_train_rollouts(batch, ["u0", "u1"], _Tokenizer(), str(tmp_path), global_step=7)

    assert path == str(tmp_path / "dumped_data" / "global_step_7_train_rollouts.jsonl")
    assert not (tmp_path / "dumped_data" / "global_step_7_train_rollouts.jsonl.tmp").exists()
    records = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert records == [
        {
            "step": 7,
            "uid": "u0",
            "trajectory_id": None,
            "prompt": "1 2",
            "response": "4 5 6",
            "reward": 1.0,
            "unshaped_reward": 1.0,
            "stop_reason": "stop",
            "response_len": 3,
            "is_last_step": None,
        },
        {
            "step": 7,
            "uid": "u1",
            "trajectory_id": None,
            "prompt": "3",
            "response": "7",
            "reward": 0.25,
            "unshaped_reward": 0.0,
            "stop_reason": "length",
            "response_len": 1,
            "is_last_step": None,
        },
    ]
