"""Config-hygiene DEFAULTS for the harbor terminus-2 agent config.

These assert that a terminal_bench yaml which OMITS the hygiene keys still gets
the safe RL defaults (recording off, raw trajectory content on), and that an
explicit yaml value still OVERRIDES the default in both directions (no falsy
`or default` bug that would silently re-enable recording).

Regression guard for the r5 engine-starvation investigation
(agent_logs/2026-07-03_r5_engine_starvation_rootcause.md).
"""

import os
import sys

import pytest
from omegaconf import OmegaConf

# The builder lives under examples/ (not an installed package).
_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

# The builder pulls in the harbor/terminal_bench agentic-RL stack, which the CPU
# dev extra deliberately does not install. Skip the module where it is absent
# (it still runs in the agentic RL env where harbor is present).
try:
    from terminal_bench.harbor_config import (  # noqa: E402
        AGENT_SCHEMA,
        ERROR_HANDLING_SCHEMA,
        ENVIRONMENT_SCHEMA,
        HarborConfigBuilder,
    )
except ImportError:
    pytest.skip("harbor deps unavailable (agentic RL extra not installed)", allow_module_level=True)


def _agent_kwargs(harbor_cfg: dict) -> dict:
    cfg = OmegaConf.create({"harbor": harbor_cfg})
    _, kwargs = HarborConfigBuilder(cfg)._build_agent_fields()
    return kwargs


def _environment_config(harbor_cfg: dict):
    cfg = OmegaConf.create({"harbor": harbor_cfg})
    return HarborConfigBuilder(cfg)._build_environment_config()


def test_schema_defaults_are_hygienic():
    assert AGENT_SCHEMA.fields["record_terminal_session"].default is False
    assert AGENT_SCHEMA.fields["trajectory_config"].default == {"raw_content": True}
    assert ERROR_HANDLING_SCHEMA.fields["fail_on_infrastructure_error"].default is False


def test_omitted_keys_get_defaults():
    kwargs = _agent_kwargs({"name": "terminus-2", "n_concurrent_trials": 8})
    assert kwargs["record_terminal_session"] is False
    assert kwargs["trajectory_config"] == {"raw_content": True}


def test_yaml_can_override_recording_on():
    kwargs = _agent_kwargs({"name": "terminus-2", "record_terminal_session": True})
    assert kwargs["record_terminal_session"] is True


def test_yaml_false_is_honored_no_falsy_bug():
    # The r5 case: explicit `false` must NOT be swallowed by the default.
    kwargs = _agent_kwargs({"name": "terminus-2", "record_terminal_session": False})
    assert kwargs["record_terminal_session"] is False


def test_apptainer_bridge_fields_are_forwarded_as_environment_kwargs():
    assert ENVIRONMENT_SCHEMA.fields["bridge_url"].harbor_field == "bridge_url"
    assert ENVIRONMENT_SCHEMA.fields["sif_cache"].harbor_field == "sif_cache"

    environment = _environment_config(
        {
            "name": "terminus-2",
            "environment_type": "apptainer",
            "bridge_url": "http://10.128.1.2:9928",
            "sif_cache": "/p/scratch/synthlaion/lee27/r2egym_sif",
        }
    )

    assert environment.type.value == "apptainer"
    assert environment.kwargs == {
        "bridge_url": "http://10.128.1.2:9928",
        "sif_cache": "/p/scratch/synthlaion/lee27/r2egym_sif",
        "auto_snapshot": False,
    }


def test_opencode_fields_reach_final_agent_config():
    cfg = OmegaConf.create(
        {
            "harbor": {
                "name": "opencode",
                "version": "1.18.8",
                "preinstalled": True,
                "prompt_template_path": "/tmp/opencode_prompt.md.j2",
                "opencode_config": {
                    "autoupdate": False,
                    "compaction": {"auto": False},
                },
            },
            "model_info": {
                "max_input_tokens": 28672,
                "max_output_tokens": 4096,
            },
        }
    )

    trial = HarborConfigBuilder(cfg).build_trial_config(
        task_path="/tmp/task",
        trials_dir="/tmp/trials",
        model_name="hosted_vllm/qwen3-6-35b-a3b-r2egym",
        api_base="http://127.0.0.1:8000/v1",
        session_id="session-1",
    )

    assert trial.agent.name == "opencode"
    assert trial.agent.kwargs["version"] == "1.18.8"
    assert trial.agent.kwargs["preinstalled"] is True
    assert trial.agent.kwargs["prompt_template_path"] == "/tmp/opencode_prompt.md.j2"
    assert trial.agent.kwargs["opencode_config"] == {
        "autoupdate": False,
        "compaction": {"auto": False},
    }
    assert trial.agent.kwargs["model_info"]["max_input_tokens"] == 28672
    assert trial.agent.kwargs["model_info"]["max_output_tokens"] == 4096
