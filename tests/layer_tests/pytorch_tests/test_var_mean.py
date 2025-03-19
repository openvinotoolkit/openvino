# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestVarMean(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

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
                super(aten_var, self).__init__()
                self.unbiased = unbiased
                self.dim = dim
                self.keepdim = keepdim
                self.op = op

            def forward(self, x):
                return self.op(x, self.dim, unbiased=self.unbiased, keepdim=self.keepdim)

        class aten_var2args(torch.nn.Module):
            def __init__(self, unbiased, op):
                super(aten_var2args, self).__init__()
                self.unbiased = unbiased
                self.op =  op
            def forward(self, x):
                return self.op(x, self.unbiased)

        ref_net = None
        op_name = f"aten::{op_type}"
        if two_args_case:
            return aten_var2args(unbiased, op), ref_net, op_name
        return aten_var(dim, unbiased, keepdim, op), ref_net, op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("unbiased", [True, False])
    @pytest.mark.parametrize("op_type", ["var", "var_mean", "std", "std_mean"])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_op2args(self, unbiased, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(unbiased, op_type=op_type), ie_device, precision, ir_version)

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
        self._test(*self.create_model(unbiased, dim, keepdim, two_args_case=False, op_type=op_type), ie_device, precision, ir_version)
