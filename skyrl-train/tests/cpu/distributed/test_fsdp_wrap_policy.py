import pytest
import torch.nn as nn

from skyrl_train.distributed.fsdp_utils import get_fsdp_wrap_policy


class PresentDecoderLayer(nn.Module):
    pass


class TextOnlyModel(nn.Module):
    _no_split_modules = ["PresentDecoderLayer", "MissingVisionBlock"]

    def __init__(self):
        super().__init__()
        self.layer = PresentDecoderLayer()


def test_wrap_policy_ignores_advertised_optional_class_absent_from_instance():
    model = TextOnlyModel()

    with pytest.warns(UserWarning, match="MissingVisionBlock"):
        policy = get_fsdp_wrap_policy(model)

    assert policy is not None


def test_wrap_policy_still_fails_when_no_requested_class_exists():
    model = TextOnlyModel()

    with pytest.raises(Exception, match="Could not find any transformer layer class"):
        get_fsdp_wrap_policy(
            model,
            {"transformer_layer_cls_to_wrap": ["MissingVisionBlock"]},
        )
