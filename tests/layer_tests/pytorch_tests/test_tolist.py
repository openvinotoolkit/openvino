import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class PrimTolist(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.tolist()  # This will trigger `prim::tolist`

class TestPrimTolist(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 3).astype(np.float32), )  # Sample tensor input

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_prim_tolist(self, ie_device, precision, ir_version):
        self._test(PrimTolist(), None, "prim::tolist", ie_device, precision, ir_version)