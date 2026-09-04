"""
Tests for the upstream-SkyRL ports of the ``dppo`` policy loss and the ``prompt_mean`` loss reduction.

uv run --frozen pytest skyrl-train/tests/cpu/algorithms/test_dppo_prompt_mean.py
"""

import math
from types import SimpleNamespace

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from skyrl_train.utils.algorithm_registry import (
    PolicyLossRegistry,
    policy_loss_requires_rollout_logprobs,
    rollout_logprobs_enabled,
)
from skyrl_train.utils.loss_reduction import (
    PRESCALED_SUM_LOSS_REDUCTIONS,
    PROMPT_MEAN_LOSS_REDUCTION,
    SUPPORTED_LOSS_REDUCTIONS,
    compute_prompt_mean_advantage_scale,
    reduce_loss,
)
from skyrl_train.utils.policy_losses import LossScaling, compute_policy_objective

# ---------------------------------------------------------------------------
# dppo
# ---------------------------------------------------------------------------


def _dppo_config(
    dppo_type: str = "binary_tv",
    delta_low: float = 0.2,
    delta_high: float = 0.2,
    loss_reduction: str = "token_mean",
    use_tis: bool = False,
) -> DictConfig:
    return DictConfig(
        {
            "policy_loss_type": "dppo",
            "loss_reduction": loss_reduction,
            "max_seq_len": 8,
            "use_tis": use_tis,
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "dppo": {"dppo_type": dppo_type, "delta_low": delta_low, "delta_high": delta_high},
        }
    )


def _tv_case():
    """Six tokens against a 0.5-probability behavior policy: A sign x divergence direction/magnitude."""
    probs = torch.tensor([[0.8, 0.6, 0.2, 0.4, 0.2, 0.8]])
    advantages = torch.tensor([[1.0, 1.0, -1.0, -1.0, 1.0, -1.0]])
    log_probs = torch.log(probs).requires_grad_(True)
    rollout_logprobs = torch.full_like(probs, math.log(0.5))
    old_log_probs = torch.zeros_like(probs)
    return log_probs, old_log_probs, advantages, rollout_logprobs


@pytest.mark.parametrize(
    "delta_low, delta_high, expected_mask",
    [
        # d = p - 0.5 = [+.3, +.1, -.3, -.1, -.3, +.3]
        # t0: A>0, d=+.3 > delta_high     -> masked
        # t1: A>0, d=+.1 <= delta_high    -> kept
        # t2: A<0, -d=+.3 > delta_low     -> masked
        # t3: A<0, -d=+.1 <= delta_low    -> kept
        # t4: A>0 but d<0 (wrong side)    -> kept
        # t5: A<0 but d>0 (wrong side)    -> kept
        (0.2, 0.2, [0.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
        # asymmetric thresholds: only delta_high gates positive advantages, only delta_low the negative ones
        (0.2, 0.35, [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
        (0.35, 0.2, [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    ],
)
def test_dppo_binary_tv_mask_value_and_gradient(delta_low, delta_high, expected_mask):
    log_probs, old_log_probs, advantages, rollout_logprobs = _tv_case()
    config = _dppo_config(delta_low=delta_low, delta_high=delta_high)
    loss_mask = torch.ones_like(advantages)

    loss, metrics = PolicyLossRegistry.get("dppo")(
        log_probs, old_log_probs, advantages, config, loss_mask=loss_mask, rollout_logprobs=rollout_logprobs
    )
    loss.backward()

    mask = torch.tensor([expected_mask])
    ratio = torch.exp(log_probs.detach() - rollout_logprobs)
    expected_loss = (-(ratio * advantages * mask)).sum() / loss_mask.sum()
    torch.testing.assert_close(loss, expected_loss)
    # The ratio is not detached: d/dlogp of -(ratio*A*m) is -(ratio*A*m); masked tokens get no gradient.
    torch.testing.assert_close(log_probs.grad, -(ratio * advantages * mask) / loss_mask.sum())
    assert torch.equal((log_probs.grad != 0).float(), mask)
    assert metrics["ppo_clip_ratio"] == pytest.approx((mask == 0).float().mean().item())


def test_dppo_uses_rollout_logprobs_not_old_logprobs():
    log_probs, _, advantages, rollout_logprobs = _tv_case()
    config = _dppo_config()
    loss_fn = PolicyLossRegistry.get("dppo")

    loss_a, _ = loss_fn(log_probs, torch.zeros_like(advantages), advantages, config, rollout_logprobs=rollout_logprobs)
    loss_b, _ = loss_fn(log_probs, torch.randn_like(advantages), advantages, config, rollout_logprobs=rollout_logprobs)
    torch.testing.assert_close(loss_a, loss_b)

    loss_c, _ = loss_fn(
        log_probs, torch.zeros_like(advantages), advantages, config, rollout_logprobs=rollout_logprobs - 0.5
    )
    assert not torch.allclose(loss_a, loss_c)


def test_dppo_ratio_is_clamped_at_plus_minus_20_in_log_space():
    # rollout logprob far below the current one: exp(30) would overflow bf16-ish ranges; clamp to exp(20).
    log_probs = torch.zeros((1, 1))
    rollout_logprobs = torch.full((1, 1), -30.0)
    advantages = torch.tensor([[-1.0]])  # A<0 with d>0 is never masked, so the loss exposes the raw ratio
    loss, _ = PolicyLossRegistry.get("dppo")(
        log_probs, torch.zeros((1, 1)), advantages, _dppo_config(), rollout_logprobs=rollout_logprobs
    )
    assert loss.item() == pytest.approx(math.exp(20.0), rel=1e-6)


def test_dppo_requires_rollout_logprobs():
    with pytest.raises(ValueError, match="rollout_logprobs are required"):
        PolicyLossRegistry.get("dppo")(torch.zeros((1, 1)), torch.zeros((1, 1)), torch.ones((1, 1)), _dppo_config())


def test_dppo_rejects_tis():
    with pytest.raises(ValueError, match="cannot be combined with use_tis"):
        PolicyLossRegistry.get("dppo")(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            _dppo_config(use_tis=True),
            rollout_logprobs=torch.zeros((1, 1)),
        )


def test_dppo_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown DPPO type"):
        PolicyLossRegistry.get("dppo")(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            _dppo_config(dppo_type="topk_tv"),
            rollout_logprobs=torch.zeros((1, 1)),
        )


@pytest.mark.parametrize(
    "delta, new_lp, advs, expect_mask, expect_clip_gt_zero",
    [
        # upstream `kl_masked`: large positive divergence with delta 0.05 -> some masking
        (0.05, [[-0.1, -0.95]], [[1.0, 1.0]], None, True),
        # upstream `kl_no_masking_within_delta`: tiny divergence with delta 0.5 -> nothing masked
        (0.5, [[-1.01, -0.99]], [[1.0, -1.0]], [[1.0, 1.0]], False),
    ],
)
def test_dppo_binary_kl(delta, new_lp, advs, expect_mask, expect_clip_gt_zero):
    rollout_logprobs = torch.tensor([[-1.0, -1.0]])
    log_probs = torch.tensor(new_lp)
    advantages = torch.tensor(advs)
    config = _dppo_config(dppo_type="binary_kl", delta_low=delta, delta_high=delta)

    loss, metrics = PolicyLossRegistry.get("dppo")(
        log_probs, torch.zeros_like(log_probs), advantages, config, rollout_logprobs=rollout_logprobs
    )
    if expect_mask is not None:
        ratio = torch.exp(log_probs - rollout_logprobs)
        expected = (-(ratio * advantages * torch.tensor(expect_mask))).mean()
        torch.testing.assert_close(loss, expected)
    if expect_clip_gt_zero:
        assert metrics["ppo_clip_ratio"] > 0.0
    else:
        assert metrics["ppo_clip_ratio"] == pytest.approx(0.0)


def test_dppo_is_registered_and_requires_rollout_logprobs():
    assert "dppo" in PolicyLossRegistry.list_available()
    assert policy_loss_requires_rollout_logprobs("dppo")
    assert policy_loss_requires_rollout_logprobs("behavior_clip")
    assert not policy_loss_requires_rollout_logprobs("regular")
    assert rollout_logprobs_enabled(OmegaConf.create({"use_tis": False, "policy_loss_type": "dppo"}))


def test_dppo_hydra_defaults_and_overrides_need_no_plus_prefix():
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    from skyrl_train.config.utils import CONFIG_DIR, DEFAULT_CONFIG_NAME

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        default = compose(config_name=DEFAULT_CONFIG_NAME)
        overridden = compose(
            config_name=DEFAULT_CONFIG_NAME,
            overrides=[
                "trainer.algorithm.policy_loss_type=dppo",
                "trainer.algorithm.loss_reduction=prompt_mean",
                "trainer.algorithm.dppo.delta_low=0.15",
                "trainer.algorithm.dppo.delta_high=0.15",
            ],
        )
    assert dict(default.trainer.algorithm.dppo) == {"dppo_type": "binary_tv", "delta_low": 0.2, "delta_high": 0.2}
    assert overridden.trainer.algorithm.policy_loss_type == "dppo"
    assert overridden.trainer.algorithm.loss_reduction == "prompt_mean"
    assert overridden.trainer.algorithm.dppo.delta_low == 0.15
    assert overridden.trainer.algorithm.dppo.delta_high == 0.15


def _validatable_dummy_config():
    from tests.cpu.util import example_dummy_config

    cfg = example_dummy_config()
    OmegaConf.update(
        cfg,
        "trainer",
        {
            "train_batch_size": 1,
            "policy_mini_batch_size": 1,
            "critic_mini_batch_size": 1,
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "placement": {
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": 1,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 1,
                "ref_num_nodes": 1,
                "ref_num_gpus_per_node": 1,
            },
        },
    )
    # One-GPU colocated geometry so validate_generator_cfg (which runs first) passes.
    cfg.generator.inference_engine_tensor_parallel_size = 1
    return cfg


def test_validate_cfg_dppo_forces_rollout_logprobs_and_accepts_prompt_mean():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.loss_reduction = "prompt_mean"
    cfg.generator.sampling_params.logprobs = None
    validate_cfg(cfg)
    assert cfg.generator.sampling_params.logprobs == 0
    assert cfg.trainer.algorithm.policy_loss_type == "dppo"
    assert cfg.trainer.algorithm.loss_reduction == "prompt_mean"


def test_validate_cfg_rejects_stacked_dppo_and_tis():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.use_tis = True
    with pytest.raises(ValueError, match="cannot be combined with use_tis"):
        validate_cfg(cfg)


def test_validate_cfg_rejects_unknown_dppo_type():
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import validate_cfg

    cfg = _validatable_dummy_config()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.dppo.dppo_type = "topk_kl"
    with pytest.raises(ValueError, match="dppo_type"):
        validate_cfg(cfg)


# ---------------------------------------------------------------------------
# prompt_mean
# ---------------------------------------------------------------------------


def _prompt_token_sums(scale: torch.Tensor, loss_mask: torch.Tensor, uids):
    """Total weight each prompt ends up with in the summed loss."""
    weight = (scale * loss_mask.float()).sum(dim=-1)
    totals = {}
    for uid, w in zip(uids, weight.tolist()):
        totals[uid] = totals.get(uid, 0.0) + w
    return totals


def test_prompt_mean_is_registered_as_prescaled_sum():
    assert PROMPT_MEAN_LOSS_REDUCTION in SUPPORTED_LOSS_REDUCTIONS
    assert PROMPT_MEAN_LOSS_REDUCTION in PRESCALED_SUM_LOSS_REDUCTIONS
    loss = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    torch.testing.assert_close(reduce_loss(loss, mask, "prompt_mean"), torch.tensor(8.0))
    torch.testing.assert_close(reduce_loss(loss, None, "prompt_mean"), torch.tensor(10.0))


def test_prompt_mean_weights_sum_to_one_over_num_prompts_regardless_of_length():
    # p0: three samples with 2, 1, 3 loss tokens (6 total); p1: two samples with 4 tokens each (8 total).
    uids = ["p0", "p0", "p0", "p1", "p1"]
    loss_mask = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=1, mini_batch_size_per_rank=5)

    assert scale.shape == (5, 1)
    torch.testing.assert_close(scale[:3, 0], torch.full((3,), 1.0 / (2 * 6)))
    torch.testing.assert_close(scale[3:, 0], torch.full((2,), 1.0 / (2 * 8)))
    totals = _prompt_token_sums(scale, loss_mask, uids)
    assert totals["p0"] == pytest.approx(0.5)
    assert totals["p1"] == pytest.approx(0.5)

    # A very long single-sample prompt and a prompt of three one-token samples carry equal weight.
    uids = ["long", "short", "short", "short"]
    loss_mask = torch.zeros((4, 100))
    loss_mask[0] = 1.0
    loss_mask[1:, 0] = 1.0
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=1, mini_batch_size_per_rank=4)
    totals = _prompt_token_sums(scale, loss_mask, uids)
    assert totals["long"] == pytest.approx(0.5)
    assert totals["short"] == pytest.approx(0.5)


def test_prompt_mean_sum_reduction_matches_upstream_prompt_mean_example():
    # Upstream SkyRL `test_prompt_mean`: rows [0,1] -> p0, rows [2,3] -> p1.
    # p0 token mean = (1+2+3)/3 = 2.0; p1 token mean = (5+6+7+8)/4 = 6.5; mean over prompts = 4.25.
    advantages = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
    scale = compute_prompt_mean_advantage_scale(
        loss_mask, ["p0", "p0", "p1", "p1"], dp_size=1, mini_batch_size_per_rank=4
    )
    loss = reduce_loss(advantages * scale, loss_mask, "prompt_mean")
    torch.testing.assert_close(loss, torch.tensor(4.25))


def test_prompt_mean_equals_mean_over_prompts_of_token_mean():
    torch.manual_seed(0)
    uids = ["a", "a", "b", "b", "b", "c"]
    loss_mask = (torch.rand(6, 7) > 0.3).float()
    loss_mask[:, 0] = 1.0  # every row keeps at least one token
    per_token_loss = torch.randn(6, 7)
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=1, mini_batch_size_per_rank=6)

    got = reduce_loss(per_token_loss * scale, loss_mask, "prompt_mean")
    expected = 0.0
    for uid in ("a", "b", "c"):
        rows = [i for i, u in enumerate(uids) if u == uid]
        expected += (per_token_loss[rows] * loss_mask[rows]).sum() / loss_mask[rows].sum()
    expected = expected / 3
    torch.testing.assert_close(got, expected)


def test_prompt_mean_normalizes_per_rank_optimizer_step_slice():
    uids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    loss_mask = torch.ones((8, 3))
    loss_mask[0, 1:] = 0.0  # ragged lengths inside a group

    # dp=2 -> chunks rows [0:4] and [4:8]; one optimizer step per chunk -> 2 prompts per slice.
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=2, mini_batch_size_per_rank=4)
    totals = _prompt_token_sums(scale, loss_mask, uids)
    assert all(v == pytest.approx(0.5) for v in totals.values())

    # dp=2 with two optimizer steps per chunk -> every slice holds exactly one prompt.
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=2, mini_batch_size_per_rank=2)
    totals = _prompt_token_sums(scale, loss_mask, uids)
    assert all(v == pytest.approx(1.0) for v in totals.values())

    # dp=1, one step over the whole batch -> 4 prompts.
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=1, mini_batch_size_per_rank=8)
    totals = _prompt_token_sums(scale, loss_mask, uids)
    assert all(v == pytest.approx(0.25) for v in totals.values())


def test_prompt_mean_padding_rows_get_zero_weight_and_are_not_prompts():
    uids = ["a", "a", "b", "b", "pad0", "pad1"]
    loss_mask = torch.ones((6, 2))
    loss_mask[4:] = 0.0
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=1, mini_batch_size_per_rank=6, pad_size=2)
    assert torch.equal(scale[4:], torch.zeros((2, 1)))
    totals = _prompt_token_sums(scale, loss_mask, uids)
    assert totals["a"] == pytest.approx(0.5)
    assert totals["b"] == pytest.approx(0.5)


def test_prompt_mean_zero_token_prompt_is_finite():
    uids = ["a", "a", "b", "b"]
    loss_mask = torch.ones((4, 2))
    loss_mask[2:] = 0.0
    scale = compute_prompt_mean_advantage_scale(loss_mask, uids, dp_size=1, mini_batch_size_per_rank=4)
    assert torch.isfinite(scale).all()
    torch.testing.assert_close(scale[2:, 0], torch.full((2,), 1.0 / (2 * 1)))


def test_prompt_mean_fails_loudly_on_bad_group_layout():
    ones = torch.ones((4, 2))
    with pytest.raises(ValueError, match="non-contiguous"):
        compute_prompt_mean_advantage_scale(ones, ["a", "b", "a", "b"], dp_size=1, mini_batch_size_per_rank=4)

    six = torch.ones((6, 2))
    # group of 3 straddles the dp chunk boundary (chunk_size=2)
    with pytest.raises(ValueError, match="straddles"):
        compute_prompt_mean_advantage_scale(six, ["a", "a", "a", "b", "b", "b"], dp_size=3, mini_batch_size_per_rank=2)
    # group of 3 straddles the optimizer-step boundary (mini_batch_size_per_rank=4)
    with pytest.raises(ValueError, match="straddles"):
        compute_prompt_mean_advantage_scale(six, ["a", "a", "a", "b", "b", "b"], dp_size=1, mini_batch_size_per_rank=4)
    # aligned layouts with the same uids are fine
    compute_prompt_mean_advantage_scale(six, ["a", "a", "a", "b", "b", "b"], dp_size=2, mini_batch_size_per_rank=3)
    compute_prompt_mean_advantage_scale(six, ["a", "a", "a", "b", "b", "b"], dp_size=1, mini_batch_size_per_rank=3)

    with pytest.raises(ValueError, match="uids"):
        compute_prompt_mean_advantage_scale(ones, ["a", "a", "b"], dp_size=1, mini_batch_size_per_rank=4)
    with pytest.raises(ValueError, match="dp_size"):
        compute_prompt_mean_advantage_scale(ones, ["a", "a", "b", "b"], dp_size=3, mini_batch_size_per_rank=4)


def _mask_sum_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, global_loss_denom=None
):
    del old_log_probs, advantages, config, rollout_logprobs, global_loss_denom
    return (log_probs * loss_mask).sum(), {}


def test_prompt_mean_policy_term_is_not_divided_by_accumulation_steps():
    config = OmegaConf.create(
        {
            "loss_reduction": "prompt_mean",
            "think_token_weight": 1.0,
            "use_entropy_loss": False,
            "entropy_loss_coef": 0.0,
            "use_kl_loss": True,
            "kl_loss_coef": 1.0,
            "kl_estimator_type": "k1",
            "use_tis": False,
            "tis_imp_ratio_cap": 2.0,
        }
    )
    common = {
        "old_action_log_probs": torch.zeros(1, 2),
        "base_action_log_probs": torch.zeros(1, 2),
        "advantages": torch.ones(1, 2),
        "loss_mask": torch.ones(1, 2),
        "rollout_logprobs": None,
        "response_span_tags": None,
        "token_entropy": torch.zeros(1, 2),
        "config": config,
        "policy_loss_fn": _mask_sum_policy_loss,
        "accumulation_steps": 4,
    }
    log_probs = torch.tensor([[1.0, 2.0]])
    caller = compute_policy_objective(action_log_probs=log_probs, scaling=LossScaling.CALLER, **common)
    scheduler = compute_policy_objective(action_log_probs=log_probs, scaling=LossScaling.MEGATRON_PIPELINE, **common)

    # policy term = 3.0 (pre-scaled sum); k1 kl = mean(logp - 0) = 1.5 with coef 1.0
    assert caller.policy_loss.item() == pytest.approx(3.0)
    assert caller.kl_loss.item() == pytest.approx(1.5)
    assert caller.optimization_loss.item() == pytest.approx(3.0 + 1.5 / 4)
    assert scheduler.optimization_loss.item() == pytest.approx(3.0 * 4 + 1.5)
    assert caller.unscaled_loss.item() == pytest.approx(4.5)


def test_trainer_scales_advantages_for_prompt_mean_before_dropping_uids():
    from skyrl_train.trainer import RayPPOTrainer
    from skyrl_train.training_batch import TrainingInputBatch

    def make_trainer(loss_reduction: str):
        trainer = object.__new__(RayPPOTrainer)
        trainer.cfg = OmegaConf.create(
            {
                "trainer": {
                    "policy_mini_batch_size": 2,
                    "algorithm": {
                        "loss_reduction": loss_reduction,
                        "advantage_batch_normalize": False,
                        "think_token_weight": 1.0,
                    },
                },
                "generator": {"n_samples_per_prompt": 2},
            }
        )
        trainer.policy_model = SimpleNamespace(actor_infos=[SimpleNamespace(rank=SimpleNamespace(dp_size=1))])
        return trainer

    def make_batch():
        advantages = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 0.0], [3.0, 0.0, 0.0], [4.0, 4.0, 0.0]])
        loss_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 0]])
        batch = TrainingInputBatch(
            {
                "advantages": advantages.clone(),
                "loss_mask": loss_mask,
                "response_mask": loss_mask.clone(),
                "rewards": torch.zeros_like(advantages),
            }
        )
        batch.metadata = {"uids": ["p0", "p0", "p1", "p1"], "pad_size": 0}
        return batch, advantages, loss_mask

    batch, advantages, loss_mask = make_batch()
    out = make_trainer("prompt_mean").finalize_advantages_for_training(batch)
    assert "uids" not in out.metadata
    assert "rewards" not in out
    # p0 has 5 loss tokens (advantage sum 7), p1 has 3 (sum 11); two prompts in the one optimizer-step slice.
    expected = advantages.clone()
    expected[:2] /= 2 * 5
    expected[2:] /= 2 * 3
    torch.testing.assert_close(out["advantages"], expected)
    # Summing the (already scaled) advantages over loss tokens gives the prompt mean of the raw advantages.
    torch.testing.assert_close(
        reduce_loss(out["advantages"], loss_mask, "prompt_mean"),
        torch.tensor((7.0 / 5 + 11.0 / 3) / 2),
    )

    # Any other reduction leaves the advantages byte-identical (no driver-side scaling).
    batch, advantages, _ = make_batch()
    out = make_trainer("token_mean").finalize_advantages_for_training(batch)
    assert torch.equal(out["advantages"], advantages)
