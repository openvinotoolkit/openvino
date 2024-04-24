# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestOuter(PytorchLayerTest):
    def _prepare_input(self, x_shape, y_shape, x_dtype, y_dtype, out=False):
        import numpy as np
        x = np.random.randn(*x_shape).astype(x_dtype)
        y = np.random.randn(*y_shape).astype(y_dtype)
        if not out:
            return (x, y)
        out = np.zeros((x_shape[0], y_shape[0]))
        return (x, y, out)

    def create_model(self, out=False, x_dtype="float32", y_dtype="float32"):
        import torch

        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32
        }
        x_dtype = dtypes[x_dtype]
        y_dtype = dtypes[y_dtype]
        class aten_outer(torch.nn.Module):
            def __init__(self, out, x_dtype, y_dtype) -> None:
                super().__init__()
                self.x_dtype = x_dtype
                self.y_dtype = y_dtype
                if out:
                    self.forward = self.forward_out

            def forward(self, x, y):
                return torch.outer(x.to(self.x_dtype), y.to(self.y_dtype))
    
            def forward_out(self, x, y, out):
                return torch.outer(x.to(self.x_dtype), y.to(self.y_dtype), out=out), out

        ref_net = None

        return aten_outer(out, x_dtype, y_dtype), ref_net, 'aten::outer'

    @pytest.mark.parametrize("x_shape", ([1], [2], [3]))
    @pytest.mark.parametrize("y_shape", ([1], [7], [5]))
    @pytest.mark.parametrize("x_dtype", ("float32", "float64", "int32"))
    @pytest.mark.parametrize("y_dtype", ("float32", "float64", "int32"))
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_numel(self, x_shape, y_shape, x_dtype, y_dtype, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out, x_dtype, y_dtype), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": out, "x_shape": x_shape, "y_shape": y_shape, "x_dtype": x_dtype, "y_dtype": y_dtype})
