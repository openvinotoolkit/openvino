# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from typing import List

from pytorch_layer_test_class import PytorchLayerTest


class TestReverse(PytorchLayerTest):
    def _prepare_input(self):
        return (np.arange(6, dtype=np.float32).reshape(2, 3),)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("length", [0, 1, 3])
    def test_reverse(self, length, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            __constants__ = ["length"]

            def __init__(self):
                super().__init__()
                self.length = length

            def forward(self, x):
                values = torch.jit.annotate(List[torch.Tensor], [])
                if self.length >= 1:
                    values.append(x + 1)
                if self.length >= 3:
                    values.append(x + 2)
                    values.append(x + 3)
                values.reverse()
                values.append(x + 4)
                return torch.cat(values, dim=0)

        self._test(Model(), "aten::reverse", ie_device, precision, ir_version, freeze_model=False)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_tensor_alias(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                first = x.clone()
                values = [first, x + 2, x + 3]
                values.reverse()
                first.add_(5)
                return values[0], values[1], values[2]

        self._test(Model(), ["aten::reverse", "aten::add_"],
                   ie_device, precision, ir_version, freeze_model=False)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_integer_list(self, ie_device, precision, ir_version):
        class Model(torch.nn.Module):
            def forward(self, x):
                shape = [x.size(0), x.size(1)]
                shape.reverse()
                return x.reshape(shape)

        self._test(Model(), "aten::reverse", ie_device, precision, ir_version, freeze_model=False)
