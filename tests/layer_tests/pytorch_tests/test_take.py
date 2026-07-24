# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestTake(PytorchLayerTest):
    def _prepare_input(self, out=False):
        inp = self.random.randn(4, 5).astype(np.float32)
        index = self.random.randint(0, 20, (3, 2)).astype(np.int64)
        if out:
            return (inp, index, np.zeros((3, 2), dtype=np.float32))
        return (inp, index)

    def create_model(self, out):
        import torch

        class aten_take(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x, index):
                return torch.take(x, index)

            def forward_out(self, x, index, out):
                return torch.take(x, index, out=out), out

        return aten_take(out), "aten::take"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_take(self, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": out})
