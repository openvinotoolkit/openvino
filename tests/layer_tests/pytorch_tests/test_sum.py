# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestSum(PytorchLayerTest):
    def _prepare_input(self, out=False, input_dtype="float32", out_dtype="float32"):
        # This test had sporadically failed by accuracy. Try to resolve that by using int numbers in input
        import numpy as np
        min_value = -10 if input_dtype not in ["uint8", "bool"] else 0
        max_value = 10 if input_dtype != "bool" else 2
        input = np.random.randint(min_value, max_value, (1, 3, 5, 5)).astype(input_dtype)
        if not out:
            return (input, )
        if out_dtype is None:
            out_dtype = input_dtype if input_dtype not in ["uint8", "bool"] else "int64"
        out = np.zeros((1, 3, 5, 5), dtype=out_dtype)
        return input, out

    def create_model(self, axes, keep_dims, out, dtype, input_dtype):

        import torch

        dtype_mapping = {
            "bool": torch.bool,
            "uint8": torch.uint8,
            "float32": torch.float32,
            "int64": torch.int64
        }
        torch_dtype = dtype_mapping[dtype] if dtype is not None else None
        input_torch_dtype = dtype_mapping[input_dtype]

        class aten_sum(torch.nn.Module):
            def __init__(self, input_dtype, axes=None, keep_dims=None, dtype=None, out=None):
                super(aten_sum, self).__init__()
                self.axes = axes
                self.keep_dims = keep_dims
                self.dtype = dtype
                self.out = out
                self.input_dtype = input_dtype
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                x = x.to(self.input_dtype)
                if self.axes is None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.sum(x)
                    else:
                        return torch.sum(x, dtype=self.dtype)
                if self.axes is not None and self.keep_dims is None:
                    if self.dtype is None:
                        return torch.sum(x, self.axes)
                    else:
                        return torch.sum(x, self.axes, dtype=self.dtype)

                if self.dtype is not None:
                    return torch.sum(x, self.axes, self.keep_dims, dtype=self.dtype)
                else:
                    return torch.sum(x, self.axes, self.keep_dims)

            def forward_out(self, x, out):
                x = x.to(self.input_dtype)
                if self.axes is None and self.keep_dims is None:
                     if self.dtype is None:
                        return torch.sum(x, out=out), out
                     else:
                         return torch.sum(x, out=out, dtype=self.dtype), out
                if self.axes is None and self.keep_dims is None:
                    if self.dtype is not None:
                        return torch.sum(x, dtype=self.dtype, out=out), out
                    else:
                        return torch.sum(x, out=out), out
                if self.axes is not None and self.keep_dims is None:
                    if self.dtype is not None:
                        return torch.sum(x, self.axes, dtype=self.dtype, out=out), out
                    else:
                        return torch.sum(x, self.axes, out=out), out

                if self.dtype is not None:
                    return torch.sum(x, self.axes, self.keep_dims, dtype=self.dtype, out=out), out
                else:
                    return torch.sum(x, self.axes, self.keep_dims, out=out), out

        ref_net = None

        return aten_sum(input_torch_dtype, axes, keep_dims, torch_dtype, out), ref_net, "aten::sum"

    @pytest.mark.parametrize("axes,keep_dims",
                             [(None, None), (None, False), (-1, None), (1, None), ((2, 3), False), ((3, 2), True)])
    @pytest.mark.parametrize("dtype", [None, "float32", "int64"])
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    @pytest.mark.parametrize("input_dtype", ["float32", "uint8", "bool", "int64"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_sum(self, axes, keep_dims, out, dtype, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(axes, keep_dims, out, dtype, input_dtype),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": out, "input_dtype": input_dtype, "out_dtype": dtype}
                   )
