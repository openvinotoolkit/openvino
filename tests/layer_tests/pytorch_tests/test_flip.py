# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestFlip(PytorchLayerTest):
    def _prepare_input(self, out=False, dtype="float32"):
        import numpy as np
        x = np.random.randn(2, 3, 4, 5).astype(dtype)
        if not out:
            return (x,)
        return (x, np.zeros_like(x).astype(dtype))


    def create_model(self, axis, out):
        import torch
        class aten_flip(torch.nn.Module):
            def __init__(self, dim, out):
                super(aten_flip, self).__init__()
                self.dim = dim
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.flip(x, self.dim)
            
            def forward_out(self, x, y):
                return torch.flip(x, self.dim, out=y), y

        ref_net = None

        return aten_flip(axis, out), ref_net, "aten::flip"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("axis", [[0], [1], [-1], [1, 2], [2, 3], [1, 2, 3]])
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8"])
    def test_flip(self, axis, out, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(axis, out), ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "dtype": dtype})
