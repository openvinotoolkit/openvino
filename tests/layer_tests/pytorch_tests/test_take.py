# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestTake(PytorchLayerTest):
    def _prepare_input(self, m, n, out=False):
        import numpy as np
        total_elements = m * n
        index = np.random.randint(0, total_elements, (n,), dtype=np.int64)
        inp = np.random.randn(m, n)
        if out:
            out_tensor = np.zeros_like(index, dtype=inp.dtype)
            return (inp, index, out_tensor)
        return (inp, index)

    def create_model(self, out):
        import torch

        class aten_take(torch.nn.Module):
            def __init__(self, out=False):
                super().__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x, index):
                return torch.take(x, index)

            def forward_out(self, x, index, out):
                return torch.take(x, index, out=out)

        return aten_take(out), "aten::take"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("m", [2, 10, 100])
    @pytest.mark.parametrize("n", [2, 10, 100])
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    def test_take(self, m, n, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version, kwargs_to_prepare_input={
            "m": m, "n": n, "out": out
        })
