"""Fast tests for concurrent eval-chunk collection in evaluate.py.

No model inference: the runner is a stub whose run() just awaits a handshake,
so the tests measure scheduling behavior (concurrent vs sequential) directly.
"""

import asyncio

import pytest
from omegaconf import OmegaConf

import skyrl_train.evaluate as evaluate_mod
from skyrl_train.evaluate import _collect_evaluation_rollouts, _WholeTrajectoryAccumulator


class _StubRunner:
    """Trajectory-runner stub tracking in-flight concurrency of run() calls."""

    def __init__(self, supports_concurrent_eval: bool):
        if supports_concurrent_eval:
            self.supports_concurrent_eval = True
        self.in_flight = 0
        self.max_in_flight = 0
        self.session_events: list = []
        self.batches: list = []

    def set_trajectory_sink(self, sink):
        pass

    async def start_eval_session(self, *, run_name, eval_step, val_set_name=None):
        self.session_events.append("start")

    async def stop_eval_session(self):
        self.session_events.append("stop")

    async def run(self, request):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        # Yield twice so overlapping calls can actually interleave.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.in_flight -= 1
        batch = {"chunk": request["chunk"]}
        self.batches.append(batch)
        return batch


def _cfg():
    return OmegaConf.create(
        {
            "trainer": {"run_name": "t"},
            "generator": {
                "eval_n_samples_per_prompt": 2,
                "backend": "vllm",
                "eval_sampling_params": {},
            },
            "environment": {"env_class": "terminal_bench"},
        }
    )


def _patch_request_prep(monkeypatch):
    calls = {"n": 0}

    def fake_prepare(prompts, n_samples, sampling_params, env_class, mode, global_step):
        i = calls["n"]
        calls["n"] += 1
        return {"chunk": i, "prompts": prompts}, [f"uid{i}"]

    monkeypatch.setattr(evaluate_mod, "prepare_trajectory_request", fake_prepare)
    monkeypatch.setattr(evaluate_mod, "get_sampling_params_for_backend", lambda backend, sp: sp)

    class _NoRecord(_WholeTrajectoryAccumulator):
        def record(self, request, batch, uids):
            self.uids.extend(uids)
            self.env_extras.append({"chunk": batch["chunk"]})

    return _NoRecord([], [], [])


@pytest.mark.asyncio
async def test_concurrent_runner_gets_overlapping_chunks(monkeypatch):
    accumulator = _patch_request_prep(monkeypatch)
    runner = _StubRunner(supports_concurrent_eval=True)
    dataloader = [["p0"], ["p1"], ["p2"], ["p3"]]

    monkeypatch.setattr(
        evaluate_mod,
        "concatenate_trajectory_batches",
        lambda batches, tis_lcs_alert_threshold=None: {"n": len(batches)},
    )
    cfg = _cfg()
    cfg.trainer.algorithm = {"tis_lcs_alert_threshold": 0.0}
    rollouts = await _collect_evaluation_rollouts(
        dataloader, runner, cfg, global_step=0, sink=None, val_set_name=None, accumulator=accumulator
    )

    assert runner.max_in_flight > 1, "chunks should overlap for a concurrent-capable runner"
    assert runner.session_events == ["start", "stop"]
    # Results recorded in submission order regardless of completion order.
    assert [e["chunk"] for e in accumulator.env_extras] == [0, 1, 2, 3]
    assert accumulator.uids == ["uid0", "uid1", "uid2", "uid3"]
    assert rollouts.batch == {"n": 4}


@pytest.mark.asyncio
async def test_plain_runner_stays_sequential(monkeypatch):
    accumulator = _patch_request_prep(monkeypatch)
    runner = _StubRunner(supports_concurrent_eval=False)
    dataloader = [["p0"], ["p1"], ["p2"]]

    monkeypatch.setattr(
        evaluate_mod,
        "concatenate_trajectory_batches",
        lambda batches, tis_lcs_alert_threshold=None: {"n": len(batches)},
    )
    cfg = _cfg()
    cfg.trainer.algorithm = {"tis_lcs_alert_threshold": 0.0}
    await _collect_evaluation_rollouts(
        dataloader, runner, cfg, global_step=0, sink=None, val_set_name=None, accumulator=accumulator
    )

    assert runner.max_in_flight == 1, "runners without the capability keep the sequential contract"
    assert runner.session_events == ["start", "stop"]


@pytest.mark.asyncio
async def test_stop_eval_session_called_on_chunk_failure(monkeypatch):
    accumulator = _patch_request_prep(monkeypatch)
    runner = _StubRunner(supports_concurrent_eval=True)

    async def failing_run(request):
        raise RuntimeError("boom")

    runner.run = failing_run
    with pytest.raises(RuntimeError):
        await _collect_evaluation_rollouts(
            [["p0"]], runner, _cfg(), global_step=0, sink=None, val_set_name=None, accumulator=accumulator
        )
    assert runner.session_events == ["start", "stop"]
