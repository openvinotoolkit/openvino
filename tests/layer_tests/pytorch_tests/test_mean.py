# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestMean(PytorchLayerTest):
    def _prepare_input(self, out, keep_dim, axis, dtype):
        import numpy as np
        if not out:
            return (np.random.randint(-10, 10, (1, 3, 224, 224)).astype(np.float32),)
        inp = np.random.randint(-10, 10, (1, 3, 224, 224)).astype(np.float32)
        calc_inp = inp.astype(dtype) if dtype is not None else inp
        if axis is None:
            out = np.mean(calc_inp, keepdims=keep_dim or False)
        else:
            out = np.mean(calc_inp, keepdims=keep_dim or False, axis=axis)
        out_tensor = np.zeros_like(out)
        return (inp, out_tensor)

    def create_model(self, axes, keep_dims, dtype, out):

        import torch

        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "uint8": torch.uint8
        }
        pt_dtype = dtypes.get(dtype)


        class aten_mean(torch.nn.Module):
            def __init__(self, axes=None, keep_dims=None, dtype=None, out=False):
                super(aten_mean, self).__init__()
                self.axes = axes
                self.keep_dims = keep_dims
                self.dtype = dtype
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                if self.axes is None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.mean(x, dtype=self.dtype)
                    return torch.mean(x)
                if self.axes is not None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.mean(x, self.axes)
                    return torch.mean(x, self.axes, dtype=self.dtype)
                if self.dtype is None:
                    return torch.mean(x, self.axes, self.keep_dims)
                return torch.mean(x, self.axes, self.keep_dims, dtype=self.dtype)

            def forward_out(self, x, out):
                if self.axes is not None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.mean(x, self.axes, out=out)
                    return torch.mean(x, self.axes, dtype=self.dtype, out=out)
                if self.dtype is None:
                    return torch.mean(x, self.axes, self.keep_dims, out=out)
                return torch.mean(x, self.axes, self.keep_dims, dtype=self.dtype, out=out)

        ref_net = None

        return aten_mean(axes, keep_dims, pt_dtype, out), ref_net, "aten::mean"

    @pytest.mark.parametrize("axes,keep_dim,dtype,out",
                             [
                                (None, None, None, False), (None, None, "float64", False), (None, None, "float32", False), (None, None, "int32", False),
                                (0, False, None, False), (0, False, None, True), (0, True, None, False), (0, True, None, True), (0, True, "float64", False),
                                (-1, None, "float32", False), (-1, None, "float32", True), (-1, True, None, False),
                                (1, None, None, False), (1, None, None, True), ((2, 3), False, None, False), ((3, 2), True, None, False)
                                ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_sum(self, axes, keep_dim, dtype, out, ie_device, precision, ir_version):
        if PytorchLayerTest.use_torch_export() and out:
            pytest.skip(reason="export fails for out")
        self._test(*self.create_model(axes, keep_dim, dtype, out),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out, "axis": axes, "dtype": dtype, "keep_dim": keep_dim})
