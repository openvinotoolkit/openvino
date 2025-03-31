# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class aten_pop(torch.nn.Module):
    def __init__(self, pop_index: int = -1):
        super().__init__()
        self.pop_index = pop_index

    def forward(self, x):
        a = torch.tensor(1, dtype=x.dtype)
        b = torch.tensor(2, dtype=x.dtype)
        lst = [a, b]
        popped = lst.pop(self.pop_index)
        return popped.reshape(())


class aten_pop_out(torch.nn.Module):
    def __init__(self, pop_index: int = -1):
        super().__init__()
        self.pop_index = pop_index

    def forward(self, x):
        a = torch.tensor(1, dtype=x.dtype)
        b = torch.tensor(2, dtype=x.dtype)
        lst = [a, b]
        popped = lst.pop(self.pop_index)
        return popped.reshape(()), torch.tensor(self.pop_index, dtype=torch.int64)


class TestPop(PytorchLayerTest):
    def _prepare_input(self):
        data = np.random.randn(2, 3).astype(np.float32)
        return (data,)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("pop_index", [-1, 0])
    def test_pop_no_out(self, pop_index, ie_device, precision, ir_version):
        model = aten_pop(pop_index=pop_index)
        self._test(model, None, "aten::pop", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("pop_index", [-1, 0])
    def test_pop_with_out(self, pop_index, ie_device, precision, ir_version):
        model = aten_pop_out(pop_index=pop_index)
        self._test(model, None, "aten::pop", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={})
