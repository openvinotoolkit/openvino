# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestForkWait(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(10, 20),)

    def create_model(self):

        class AddMod(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: int):
                return a + b, a - b

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = AddMod()

            def forward(self, input):
                fut = torch.jit.fork(self.mod, a=input, b=2)
                return torch.jit.wait(fut)

        return Mod(), None, ["prim::fork", "aten::wait"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("to_trace"), [True, False])
    def test_fork_wait(self, to_trace, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=to_trace)
