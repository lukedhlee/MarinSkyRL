"""
Main entrypoint for generating rollouts on terminal bench tasks.
"""

import os
import signal
import sys
import ray
import asyncio
import hydra
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import (
    create_ray_wrapped_inference_engines_from_config,
    create_remote_inference_engines_from_config,
    BasePPOExp,
    config_dir,
)
from skyrl_train.generators.utils import prepare_generator_input
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from examples.terminal_bench.terminal_bench_generator import TerminalBenchGenerator
from examples.terminal_bench.dataset import TerminalBenchTaskDataset


class TerminalBenchGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
        return TerminalBenchGenerator(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,  # Pass terminal_bench config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            moe_router_replay=bool(
                cfg.trainer.policy.fsdp_config.get("moe_router_replay", False)
            ),
            use_tis=bool(cfg.trainer.algorithm.get("use_tis", False)),
            tito_full=cfg.trainer.algorithm.get("tito_full", None),
        )

    def _setup_generator(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(self.cfg, self.colocate_pg, tokenizer)
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg, tokenizer)

        inference_engine_client = InferenceEngineClient(inference_engines, tokenizer, self.cfg)
        asyncio.run(inference_engine_client.wake_up())

        return self.get_generator(self.cfg, tokenizer, inference_engine_client)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            TerminalBenchTaskDataset: The training dataset.
        """
        prompts_dataset = TerminalBenchTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def run(self):
        # Externally routed inference endpoint (bridge/sandbox deployments):
        # sandboxes on the CPU fleet reach the engines via a reverse forward
        # (e.g. 10.14.0.46:<port> -> head:8000), not via the head's loopback.
        # The trainer's fan-out path (RolloutDispatcher) applies this exact
        # override; without it the generator writes baseURL 127.0.0.1:8000
        # into every sandbox's agent config and the agents call THEMSELVES —
        # zero requests ever reach the engines (smoke 1293267).
        rollout_host = os.environ.get("SKYRL_ROLLOUT_HTTP_ENDPOINT_HOST")
        rollout_port = os.environ.get("SKYRL_ROLLOUT_HTTP_ENDPOINT_PORT")
        if rollout_host:
            self.cfg.generator.http_endpoint_host = rollout_host
            if rollout_port:
                self.cfg.generator.http_endpoint_port = int(rollout_port)
            logger.info(
                f"generate: using externally routed inference endpoint "
                f"{rollout_host}:{self.cfg.generator.http_endpoint_port}"
            )

        generator = self._setup_generator()

        # Build input from the training dataset via the SAME builder the trainer
        # uses (generators/utils.py): expands each task ×n_samples_per_prompt
        # (p@k) and attaches the trajectory_ids / batch_metadata the generator's
        # generate() requires — the hand-rolled GeneratorInput this replaces
        # provided neither (KeyError: 'trajectory_ids', and no p@k expansion).
        items = [self.train_dataset[i] for i in range(len(self.train_dataset))]
        input_batch, _ = prepare_generator_input(
            items,
            self.cfg.generator.n_samples_per_prompt,
            get_sampling_params_for_backend(self.cfg.generator.backend, self.cfg.generator.sampling_params),
            self.cfg.environment.env_class,
            "train",
            0,
        )
        logger.info(
            f"generate input: {len(items)} tasks x {self.cfg.generator.n_samples_per_prompt} samples "
            f"= {len(input_batch['prompts'])} trials"
        )

        # Start generation. startup() creates the shared QueueOrchestrator +
        # its lock — the trainer path calls it once before the first generate();
        # without it generate() dies at `async with self._orchestrator_lock`
        # (None). shutdown() flushes/stops the orchestrator cleanly.
        async def _generate():
            await generator.startup()
            try:
                await generator.generate(input_batch)
            finally:
                await generator.shutdown()

        asyncio.run(_generate())


@ray.remote(num_cpus=1, max_retries=0)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    def _sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM on head node, shutting down Ray...")
        ray.shutdown()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
    finally:
        logger.info("Shutting down Ray on head node...")
        ray.shutdown()


if __name__ == "__main__":
    main()
