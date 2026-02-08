# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest
import torch


class TestWhileLoopFX(PytorchLayerTest):
    """Tests for torch.while_loop support in FX export (torch.export)."""

    def _prepare_input(self):
        return (np.random.randn(*self.input_shape).astype(np.float32),)

    def create_model_simple_counter(self, num_iterations):
        """Simple while_loop that counts up to num_iterations."""
        class WhileLoopCounter(torch.nn.Module):
            def __init__(self, limit):
                super().__init__()
                self.limit = limit

            def forward(self, x):
                def cond(i, x):
                    return i < self.limit

                def body(i, x):
                    return i + 1, x + 1

                i = torch.tensor(0)
                _, out = torch.while_loop(cond, body, (i, x))
                return out

        return WhileLoopCounter(num_iterations), None, "while_loop"

    def create_model_accumulator(self):
        """While loop that accumulates a value - with clone to avoid aliasing."""
        class WhileLoopAccumulator(torch.nn.Module):
            def forward(self, x):
                def cond(i, acc, x):
                    return i < 5

                def body(i, acc, x):
                    # Clone x to avoid aliasing (PyTorch higher order op limitation)
                    return i + 1, acc + x, x.clone()

                i = torch.tensor(0)
                acc = torch.zeros_like(x)
                _, result, _ = torch.while_loop(cond, body, (i, acc, x))
                return result

        return WhileLoopAccumulator(), None, "while_loop"

    def create_model_scalar_only(self):
        """While loop with only scalar values (no tensor input needed)."""
        class WhileLoopScalar(torch.nn.Module):
            def forward(self, x):
                def cond(i):
                    return i < 10

                def body(i):
                    return (i + 1,)

                i = torch.tensor(0)
                (result,) = torch.while_loop(cond, body, (i,))
                # Use x to keep it as an input, but return scalar result
                return result + x.sum() * 0

        return WhileLoopScalar(), None, "while_loop"

    @pytest.mark.parametrize("num_iterations", [1, 3, 5, 10])
    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_while_loop_counter(self, num_iterations, ie_device, precision, ir_version):
        self.input_shape = (2, 4)
        self._test(*self.create_model_simple_counter(num_iterations), ie_device, precision,
                   ir_version, trace_model=False, use_torch_export=True, fx_kind="while_loop")

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_while_loop_accumulator(self, ie_device, precision, ir_version):
        self.input_shape = (3, 5)
        self._test(*self.create_model_accumulator(), ie_device, precision,
                   ir_version, trace_model=False, use_torch_export=True, fx_kind="while_loop")

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    def test_while_loop_scalar_only(self, ie_device, precision, ir_version):
        self.input_shape = (2, 3)
        self._test(*self.create_model_scalar_only(), ie_device, precision,
                   ir_version, trace_model=False, use_torch_export=True, fx_kind="while_loop")


class TestLoopWithAlias(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(*self.shape).astype(np.float32),)

    def create_model(self, n):
        import torch

        class loop_alias_model(torch.nn.Module):
            def __init__(self, n):
                super(loop_alias_model, self).__init__()
                self.n = n

            def forward(self, x):
                N = x.shape[1]
                res = torch.zeros(1, self.n, dtype=torch.long)
                d = torch.ones(1, N) * 1e10
                f = torch.zeros(1, dtype=torch.long)

                for i in range(self.n):
                    res[:, i] = f
                    _d = torch.sum((x - x[0, f, :]) ** 2, -1)
                    m = _d < d
                    d[m] = _d[m]
                    f = torch.max(d, -1)[1]
                return res

        return loop_alias_model(n), None, ["prim::Loop", "aten::copy_"]

    @pytest.mark.parametrize("s,n", [([1, 1024, 3], 512), ([1, 512, 3], 128)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_loop_alias(self, s, n, ie_device, precision, ir_version):
        self.shape = s
        self._test(*self.create_model(n), ie_device, precision,
                   ir_version, use_convert_model=True)
