import asyncio
from collections.abc import Awaitable, Callable

import pytest
import ray
from omegaconf import OmegaConf

from skyrl_train.trajectory_runners.harbor.rollout_dispatcher import (
    RolloutCoordinatorRPCTimeoutError,
    RolloutDispatcher,
)


class _RemoteMethod:
    def __init__(self, call: Callable[[], Awaitable[dict]]):
        self._call = call

    def remote(self, *_args):
        return self._call()


class _Coordinator:
    def __init__(self, call: Callable[[], Awaitable[dict]]):
        self.run_shard = _RemoteMethod(call)


@ray.remote
class _BlockingCoordinator:
    def __init__(self):
        self._release = asyncio.Event()
        self._finished = asyncio.Event()
        self._cancelled = False

    async def run_shard(self, *_args):
        try:
            await self._release.wait()
            return {"response_ids": [[1]], "rollout_metrics": {}}
        except asyncio.CancelledError:
            self._cancelled = True
            raise
        finally:
            self._finished.set()

    async def release(self):
        self._release.set()

    async def wait_for_completion(self):
        await self._finished.wait()
        return self._cancelled


def _dispatcher(actor: object, *, timeout: float) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        cfg=OmegaConf.create({}),
        trajectory_runner_cfg=OmegaConf.create({}),
        terminal_bench_cfg=OmegaConf.create({}),
        num_coordinators=1,
        cpus_per_coordinator=1,
        coordinator_rpc_timeout=timeout,
    )
    dispatcher._actors = [actor]
    return dispatcher


@pytest.mark.asyncio
async def test_coordinator_rpc_returns_trajectory_batch():
    expected = {"response_ids": [[1]], "rollout_metrics": {}}

    async def completed_rpc():
        return expected

    dispatcher = _dispatcher(_Coordinator(completed_rpc), timeout=1)

    assert await dispatcher.run({"prompts": ["task"]}) is expected


@pytest.mark.asyncio
async def test_coordinator_rpc_timeout_does_not_cancel_remote_work(ray_init):
    actor = _BlockingCoordinator.remote()
    dispatcher = _dispatcher(actor, timeout=0.1)

    with pytest.raises(RolloutCoordinatorRPCTimeoutError):
        await dispatcher.run({"prompts": ["task"]})

    await actor.release.remote()
    assert await actor.wait_for_completion.remote() is False


@pytest.mark.asyncio
async def test_coordinator_rpc_preserves_remote_timeout_error():
    remote_error = TimeoutError("remote post-processing timed out")

    async def failed_rpc():
        raise remote_error

    dispatcher = _dispatcher(_Coordinator(failed_rpc), timeout=1)

    with pytest.raises(TimeoutError) as raised:
        await dispatcher.run({"prompts": ["task"]})

    assert raised.value is remote_error


class _KwargsRemoteMethod:
    """Remote-method stub that records calls and returns an awaitable."""

    def __init__(self, log: list, name: str):
        self._log = log
        self._name = name

    def remote(self, *args, **kwargs):
        self._log.append((self._name, args, kwargs))

        async def _done():
            return {"response_ids": [[1]], "rollout_metrics": {}}

        return _done()


class _EvalCoordinator:
    def __init__(self):
        self.calls: list = []
        self.run_shard = _KwargsRemoteMethod(self.calls, "run_shard")
        self.start_eval_session = _KwargsRemoteMethod(self.calls, "start_eval_session")
        self.stop_eval_session = _KwargsRemoteMethod(self.calls, "stop_eval_session")


def _fanout_dispatcher(actors: list) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher(
        cfg=OmegaConf.create({}),
        trajectory_runner_cfg=OmegaConf.create({}),
        terminal_bench_cfg=OmegaConf.create({}),
        num_coordinators=len(actors),
        cpus_per_coordinator=1,
        coordinator_rpc_timeout=5,
    )
    dispatcher._actors = actors
    return dispatcher


@pytest.mark.asyncio
async def test_eval_session_broadcasts_to_all_coordinators():
    actors = [_EvalCoordinator() for _ in range(3)]
    dispatcher = _fanout_dispatcher(actors)

    await dispatcher.start_eval_session(run_name="r", eval_step=7, val_set_name="v")
    for actor in actors:
        starts = [c for c in actor.calls if c[0] == "start_eval_session"]
        assert len(starts) == 1
        assert starts[0][2] == {"run_name": "r", "eval_step": 7, "val_set_name": "v"}
    assert dispatcher._eval_session_active is True

    await dispatcher.stop_eval_session()
    for actor in actors:
        assert [c[0] for c in actor.calls].count("stop_eval_session") == 1
    assert dispatcher._eval_session_active is False


@pytest.mark.asyncio
async def test_eval_groups_round_robin_across_coordinators():
    actors = [_EvalCoordinator() for _ in range(3)]
    dispatcher = _fanout_dispatcher(actors)
    await dispatcher.start_eval_session(run_name="r", eval_step=0, val_set_name=None)

    for _ in range(6):
        await dispatcher.run({"prompts": ["task"]})

    per_actor = [sum(1 for c in actor.calls if c[0] == "run_shard") for actor in actors]
    assert per_actor == [2, 2, 2]


def test_dispatcher_advertises_concurrent_eval():
    dispatcher = _fanout_dispatcher([_EvalCoordinator()])
    assert dispatcher.supports_concurrent_eval is True
