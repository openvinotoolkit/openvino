# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestGather(PytorchLayerTest):
    def _prepare_input(self, m, n, max_val, out=False):
        import numpy as np
        index = np.random.randint(0, max_val, (m, n)).astype(np.int64)
        inp = np.random.randn(m, n).astype(np.float32)
        if out:
            axis = int(max_val == n)
            out = np.zeros_like(np.take(inp, index, axis))
            return (inp, index, out)
        return (inp, index)

    def create_model(self, axis, out):
        import torch

        class aten_gather(torch.nn.Module):
            def __init__(self, axis, out=False):
                super(aten_gather, self).__init__()
                self.axis = axis
                if out:
                    self.forward = self.forward_out

            def forward(self, x, index):
                return torch.gather(x, self.axis, index)

            def forward_out(self, x, index, out):
                return torch.gather(x, self.axis, index, out=out)

        ref_net = None

        return aten_gather(axis, out), ref_net, "aten::gather"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("m", [2, 10, 100])
    @pytest.mark.parametrize("n", [2, 10, 100])
    @pytest.mark.parametrize("axis", [0, 1])
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    def test_gather(self, m, n, axis, out, ie_device, precision, ir_version):
        self._test(*self.create_model(axis, out), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "m": m, "n": n, "max_val": m if axis == 0 else n, "out": out
        })
