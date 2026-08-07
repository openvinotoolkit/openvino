# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestVarMean(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 224, 224),)

    def create_model(self, unbiased, dim=None, keepdim=True, two_args_case=True, op_type="var"):
        import torch

        ops = {
            "var": torch.var,
            "var_mean": torch.var_mean,
            "std": torch.std,
            "std_mean": torch.std_mean
        }

        op = ops[op_type]

        class aten_var(torch.nn.Module):
            def __init__(self, dim, unbiased, keepdim, op):
                super().__init__()
                self.unbiased = unbiased
                self.dim = dim
                self.keepdim = keepdim
                self.op = op

            def forward(self, x):
                return self.op(x, self.dim, unbiased=self.unbiased, keepdim=self.keepdim)

        class aten_var2args(torch.nn.Module):
            def __init__(self, unbiased, op):
                super().__init__()
                self.unbiased = unbiased
                self.op =  op
            def forward(self, x):
                return self.op(x, self.unbiased)

        op_name = f"aten::{op_type}"
        if two_args_case:
            return aten_var2args(unbiased, op), op_name
        return aten_var(dim, unbiased, keepdim, op), op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("unbiased", [True, False])
    @pytest.mark.parametrize("op_type", ["var", "var_mean", "std", "std_mean"])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_op2args(self, unbiased, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(unbiased, op_type=op_type), ie_device, precision, ir_version,
                   trace_model=True)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("unbiased", [False, True])
    @pytest.mark.parametrize("dim", [None, 0, 1, 2, 3, -1, -2, (0, 1), (-1, -2), (0, 1, -1), (0, 1, 2, 3)])
    @pytest.mark.parametrize("keepdim", [True, False])
    @pytest.mark.parametrize("op_type", ["var", "var_mean", "std", "std_mean"])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_op(self, unbiased, dim, keepdim, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(unbiased, dim, keepdim, two_args_case=False, op_type=op_type), ie_device, precision, ir_version,
                   trace_model=True)


class TestVarMeanCorrection(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(8, 8, 8),)

    def create_model(self, correction, dim, keepdim, op_type):
        import torch

        ops = {
            "var": torch.var,
            "var_mean": torch.var_mean,
            "std": torch.std,
            "std_mean": torch.std_mean
        }

        op = ops[op_type]

        class aten_var_correction(torch.nn.Module):
            def __init__(self, dim, correction, keepdim, op):
                super().__init__()
                self.correction = correction
                self.dim = dim
                self.keepdim = keepdim
                self.op = op

            def forward(self, x):
                return self.op(x, self.dim, correction=self.correction, keepdim=self.keepdim)

        return aten_var_correction(dim, correction, keepdim, op), f"aten::{op_type}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("correction", [0, 1, 2, 3])
    @pytest.mark.parametrize("dim", [0, 1, -1, (0, 1), (-1, -2)])
    @pytest.mark.parametrize("keepdim", [True, False])
    # Input is 8x8x8 so the reduction factor (>= 8) stays strictly greater than
    # the largest correction, keeping the unbiased estimator well-defined.
    @pytest.mark.parametrize("op_type", ["var", "var_mean", "std", "std_mean"])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_correction(self, correction, dim, keepdim, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(correction, dim, keepdim, op_type=op_type), ie_device, precision, ir_version,
                   trace_model=True)
