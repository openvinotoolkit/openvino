# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCumSum(PytorchLayerTest):
    def _prepare_input(self, out=False, out_dtype=None):
        import numpy as np
        x = np.random.randn(1, 3, 224, 224).astype(np.float32)
        if not out:
            return (x, )
        y =  np.random.randn(1, 3, 224, 224).astype(np.float32)
        if out_dtype is not None:
            y = y.astype(out_dtype)
        return (x, y)


    def create_model(self, axis, dtype_str, out, dtype_from_input):
        import torch

        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "uint8": torch.uint8
        }

        dtype = dtypes.get(dtype_str)

        class aten_cumsum(torch.nn.Module):
            def __init__(self, axis, dtype, out=False, dtype_from_input=False):
                super(aten_cumsum, self).__init__()
                self.axis = axis
                self.dtype = dtype
                if dtype_from_input:
                    self.forward_out = self.forward_out_prim_dtype
                if out:
                    self.forward =  self.forward_out
                if self.dtype is not None:
                    if not dtype_from_input: 
                        self.forward = self.forward_dtype if not out else self.forward_out_dtype

            def forward(self, x):
                return torch.cumsum(x, self.axis)
            
            def forward_dtype(self, x):
                return torch.cumsum(x, self.axis, dtype=self.dtype)
            
            def forward_out(self, x, y):
                return y, torch.cumsum(x, self.axis, out=y)

            def forward_out_dtype(self, x, y):
                return y, torch.cumsum(x, self.axis, dtype=self.dtype, out=y)

            def forward_out_prim_dtype(self, x, y):
                return y, torch.cumsum(x, self.axis, dtype=y.dtype, out=y)

        ref_net = None

        return aten_cumsum(axis, dtype, out, dtype_from_input), ref_net, "aten::cumsum"

    @pytest.mark.parametrize("axis", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("out,dtype_from_input", [(False, False), (True, False), (True, True)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_cumsum(self, axis, dtype, out, dtype_from_input, ie_device, precision, ir_version):
        if ie_device == "GPU" and dtype == "int8":
            pytest.xfail(reason="Cumsum for i8 is unsupported on GPU")
        if out and PytorchLayerTest.use_torch_export():
            pytest.skip(reason="export fails for out")
        self._test(*self.create_model(axis, dtype, out, dtype_from_input), ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "out_dtype": dtype})
