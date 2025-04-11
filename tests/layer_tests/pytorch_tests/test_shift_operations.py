# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestShiftOperators(PytorchLayerTest):
    def _prepare_input(self, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape):
        choices = np.array([1, 2, 4, 8, 16, 32])  
        shifts = np.array([0, 1, 2, 3, 4, 5])     

        x = np.random.choice(choices, lhs_shape).astype(lhs_dtype)
        y = np.random.choice(shifts, rhs_shape).astype(rhs_dtype)
        return x, y

    def create_model(self):
        class aten_shift(torch.nn.Module):
            def forward(self, lhs, rhs):
                return lhs << rhs, lhs >> rhs 

        ref_net = None
        return aten_shift(), ref_net, ("aten::__lshift__", "aten::__rshift__")

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("lhs_dtype", ["int32", "int64"])
    @pytest.mark.parametrize("rhs_dtype", ["int32", "int64"])
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),  
            ([2, 3], []),      
            ([], [2, 3]),  
            ([], []), 
        ],
    )
    def test_shift_operators(self, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "lhs_dtype": lhs_dtype,
                "rhs_dtype": rhs_dtype,
                "lhs_shape": lhs_shape,
                "rhs_shape": rhs_shape,
            },
            trace_model=True,
            freeze_model=False,
        )


class TestBitwiseShiftFunctions(PytorchLayerTest):
    def _prepare_input(self, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape):
        choices = np.array([1, 2, 4, 8, 16, 32])  
        shifts = np.array([0, 1, 2, 3, 4, 5])    

        x = np.random.choice(choices, lhs_shape).astype(lhs_dtype)
        y = np.random.choice(shifts, rhs_shape).astype(rhs_dtype)
        return x, y

    def create_model(self):
        class aten_bitwise_shift(torch.nn.Module):
            def forward(self, lhs, rhs):
                return (
                    torch.bitwise_left_shift(lhs, rhs),
                    # torch.bitwise_right_shift(lhs, rhs) - temporarily disable
                )

        return aten_bitwise_shift(), None, ("aten::bitwise_left_shift",)  # "aten::bitwise_right_shift")

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("lhs_dtype", ["int32", "int64"])
    @pytest.mark.parametrize("rhs_dtype", ["int32", "int64"])
    @pytest.mark.parametrize(
        ("lhs_shape", "rhs_shape"),
        [
            ([2, 3], [2, 3]),  
            ([2, 3], []),      
            ([], [2, 3]),      
            ([], []), 
        ],
    )
    def test_bitwise_shift_functions(self, lhs_dtype, rhs_dtype, lhs_shape, rhs_shape, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "lhs_dtype": lhs_dtype,
                "rhs_dtype": rhs_dtype,
                "lhs_shape": lhs_shape,
                "rhs_shape": rhs_shape,
            },
            trace_model=True,
            freeze_model=False,
        )