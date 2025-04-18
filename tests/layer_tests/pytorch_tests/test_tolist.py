import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class aten_tolist(torch.nn.Module):
    def forward(self, x):
        return x.tolist()


class TestPrimTolist(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_tolist(self, ie_device, precision, ir_version):
        self._test(aten_tolist(), None, ["prim::tolist", "op::node_skip"],
                   ie_device, precision, ir_version, freeze_model=False)
