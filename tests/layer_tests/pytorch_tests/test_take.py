# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestTake(PytorchLayerTest):

    def _prepare_input(self, out=False):
        x = self.random.randn(4, 3).astype(np.float32)
        idx = np.array([0, 11, 2, 5], dtype=np.int64)
        if out:
            return (x, idx, np.zeros(4, dtype=np.float32))
        return (x, idx)

    def create_model(self, out=False):
        import torch

        class aten_take(torch.nn.Module):
            def forward(self, x, idx):
                return torch.take(x, idx)

        class aten_take_out(torch.nn.Module):
            def forward(self, x, idx, out):
                return torch.take(x, idx, out=out), out

        model = aten_take_out() if out else aten_take()
        return model, "aten::take"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [True, False])
    def test_take(self, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": out})
