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


class TestVarMeanDefaultCorrection(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 5, 7),)

    def create_model(self, op_type, dim):
        import torch

        ops = {
            "var": torch.var,
            "var_mean": torch.var_mean,
            "std": torch.std,
            "std_mean": torch.std_mean
        }

        class aten_var_default(torch.nn.Module):
            def __init__(self, op, dim):
                super().__init__()
                self.op = op
                self.dim = dim

            def forward(self, x):
                # No unbiased/correction argument: PyTorch applies Bessel correction.
                # torch.export records this as aten.<op>.correction without a correction value.
                if self.dim is None:
                    return self.op(x)
                return self.op(x, self.dim, keepdim=True)

        return aten_var_default(ops[op_type], dim), f"aten::{op_type}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("op_type", ["var", "var_mean", "std", "std_mean"])
    @pytest.mark.parametrize("dim", [None, 1, (0, 2)])
    def test_op_default_correction(self, op_type, dim, ie_device, precision, ir_version):
        self._test(*self.create_model(op_type, dim), ie_device, precision, ir_version, trace_model=True)
