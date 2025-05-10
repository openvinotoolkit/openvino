import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class aten_unchecked_cast(torch.nn.Module):
    def forward(self, x):
        return torch.ops.prim.unchecked_cast(x, dtype=torch.float32)

class TestPrimUncheckedCast(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([1, 2, 3], dtype=np.int32),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_unchecked_cast(self, ie_device, precision, ir_version):
        self._test(aten_unchecked_cast(), None, ["prim::unchecked_cast"],
                   ie_device, precision, ir_version, freeze_model=False)