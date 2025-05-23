# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestPythonOp(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3, 128, 128),)

    def create_model(self):
        import torch
        from torch.autograd.function import Function

        class _ExpF(Function):
            @staticmethod
            def forward(ctx, input_tensor):
                exp = torch.exp(input_tensor)
                ctx.save_for_backward(exp)
                return exp

        exp_f = _ExpF.apply

        class prim_pythonop(torch.nn.Module):
            def forward(self, input_tensor):
                return exp_f(input_tensor)

        ref_net = None

        return prim_pythonop(), ref_net, "prim::PythonOp"

    @pytest.mark.parametrize(
        ("use_trace"),
        [
            True,
            pytest.param(
                False,
                marks=pytest.mark.xfail(
                    reason="Subgraph of prim::PythonOp cannot be retrieved using if using scripting."
                ),
            ),
        ],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pythonop(self, use_trace, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            trace_model=use_trace
        )
