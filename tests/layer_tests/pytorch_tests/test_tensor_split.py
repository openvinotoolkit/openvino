# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Collection
from numbers import Number

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestTensorSplit(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.rand(*self.input_shape),)

    def create_model(self, splits, axis):
        class aten_tensor_split(torch.nn.Module):
            def __init__(self, splits, dim) -> None:
                super().__init__()
                self.splits = splits
                self.dim = dim
                num_outs = None
                if isinstance(splits, Number):
                    num_outs = splits
                elif isinstance(splits, Collection):
                    num_outs = len(splits) + 1
                self.forward = getattr(self, f"forward_{num_outs}")

            def forward_2(self, input_tensor):
                a, b = torch.tensor_split(input_tensor, self.splits, dim=self.dim)
                return a, b

            def forward_3(self, input_tensor):
                a, b, c = torch.tensor_split(input_tensor, self.splits, dim=self.dim)
                return a, b, c

            def forward_4(self, input_tensor):
                a, b, c, d = torch.tensor_split(input_tensor, self.splits, dim=self.dim)
                return a, b, c, d

        return aten_tensor_split(splits, axis), None, "aten::tensor_split"

    @pytest.mark.parametrize("input_shape", [(2, 1, 8), (3, 5, 7, 11)])
    @pytest.mark.parametrize(
        "splits",
        [
            # 1, Does not work for 1 - no list_unpack present in the graph
            2,
            3,
            4,
            [2],
            [5],
            [-1],
            [-5],
            [1, 3],
            [1, 3, 5],
            [5, -1, 7],
        ],
    )
    @pytest.mark.parametrize("axis", [0, 1, -1])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_tensor_split(self, input_shape, splits, axis, ie_device, precision, ir_version):
        self.input_shape = input_shape
        self._test(*self.create_model(splits, axis), ie_device, precision, ir_version)
