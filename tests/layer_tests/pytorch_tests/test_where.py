# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class Testwhere(PytorchLayerTest):
    def _prepare_input(self, mask_fill='ones', mask_dtype=bool, return_x_y=False, x_dtype="float32", y_dtype=None):
        input_shape = [2, 10]
        mask = np.zeros(input_shape).astype(mask_dtype)
        if mask_fill == 'ones':
            mask = np.ones(input_shape).astype(mask_dtype)
        if mask_fill == 'random':
            idx = np.random.choice(10, 5)
            mask[:, idx] = 1
        x = np.random.randn(*input_shape).astype(x_dtype)
        y = np.random.randn(*input_shape).astype(y_dtype or x_dtype)
        return (mask,) if not return_x_y else (mask, x, y)

    def create_model(self, as_non_zero, dtypes=None):
        import torch

        dtype_map = {
            "float32": torch.float32,
            "int32": torch.int32
        }

        torch_dtypes = None
        if dtypes:
            torch_dtypes = (dtype_map[dtypes[0]], dtype_map[dtypes[1]])

        class aten_where(torch.nn.Module):
            def __init__(self, dtypes) -> None:
                super().__init__()
                self.x_dtype = dtypes[0]
                self.y_dtype = dtypes[1]


            def forward(self, cond, x, y):
                return torch.where(cond, x.to(self.x_dtype), y.to(self.y_dtype))

        class aten_where_as_nonzero(torch.nn.Module):
            def forward(self, cond):
                return torch.where(cond)

        ref_net = None

        if as_non_zero:
            return aten_where_as_nonzero(), ref_net, "aten::where"
        return aten_where(torch_dtypes), ref_net, "aten::where"

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, bool])  # np.float32 incorrectly casted to bool
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("y_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_where(self, mask_fill, mask_dtype, x_dtype, y_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(False, dtypes=(x_dtype, y_dtype)),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill, 
                       'mask_dtype': mask_dtype, 
                       'return_x_y': True,
                       "x_dtype": x_dtype,
                       "y_dtype": y_dtype
                       })

    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, bool])  # np.float32 incorrectly casted to bool
    @pytest.mark.parametrize("x_dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_where_as_nonzero(self, mask_fill, mask_dtype, x_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(True),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       'mask_fill': mask_fill, 
                       'mask_dtype': mask_dtype, 
                       'return_x_y': False,
                       "x_dtype": x_dtype,
                       },
                   trace_model=True)
