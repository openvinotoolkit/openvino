import pytest
import torch
from torch import nn
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class aten_set_(nn.Module):
    def forward(self, x, y):
        return x.set_(y)


class TestSet(PytorchLayerTest):
    def _prepare_input(self):
        return (
            np.random.randn(1, 2, 3, 4).astype(np.float32),
            np.random.randn(1).astype(np.float32),
        )

    def create_model(self):
        return aten_set_()

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_set(self, ie_device, precision, ir_version):
        self._test(
            self.create_model(),
            None,
            "aten::set_",
            ie_device,
            precision,
            ir_version,
        )
