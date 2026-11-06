# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_pop(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x, y):
        tensors = [x, y, x + y, x * y]
        if self.idx is None:
            popped = tensors.pop()
        else:
            popped = tensors.pop(self.idx)
        return popped, torch.cat(tensors, 0)


class aten_pop_after_append(torch.nn.Module):
    def forward(self, x, y):
        tensors = [x]
        tensors.append(y)
        tensors.append(x - y)
        first = tensors.pop(0)
        last = tensors.pop()
        return first, last, torch.cat(tensors, 0)


class aten_pop_twice(torch.nn.Module):
    def forward(self, x, y):
        tensors = [x, y, x + y]
        a = tensors.pop()
        b = tensors.pop()
        return a, b, tensors[0]


class TestPop(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3).astype(np.float32),
                self.random.randn(2, 3).astype(np.float32))

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("idx", [None, 0, 1, -1, -2])
    def test_pop(self, idx, ie_device, precision, ir_version):
        self._test(aten_pop(idx), ["aten::pop", "prim::ListConstruct"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pop_after_append(self, ie_device, precision, ir_version):
        self._test(aten_pop_after_append(), ["aten::pop", "aten::append"],
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pop_twice(self, ie_device, precision, ir_version):
        self._test(aten_pop_twice(), ["aten::pop", "prim::ListConstruct"],
                   ie_device, precision, ir_version)


class aten_pop_shape(torch.nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x, y):
        shape = x.size()
        dim = shape.pop(self.idx)
        return x.reshape([dim, -1]), torch.tensor(shape)


class aten_pop_inplace(torch.nn.Module):
    def forward(self, x, y):
        tensors = [x * 2, y]
        first = tensors.pop(0)
        first.add_(1)
        return first, torch.cat(tensors, 0)


class TestPopExtra(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(2, 3, 4).astype(np.float32),
                self.random.randn(2, 3, 4).astype(np.float32))

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("idx", [0, 1, 2, -1, -2])
    def test_pop_shape(self, idx, ie_device, precision, ir_version):
        self._test(aten_pop_shape(idx), "aten::pop", ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pop_inplace(self, ie_device, precision, ir_version):
        self._test(aten_pop_inplace(), "aten::pop", ie_device, precision, ir_version)
