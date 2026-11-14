# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shape lists mixing rank-1 `y.size()` with rank-0 ints must be flattened, not unsqueezed, before Concat.

Covers flatten_list_element_for_concat() in src/frontends/pytorch/src/utils.cpp on each get_list_as_outputs() path.
"""

from typing import List

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class aten_view_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        return x.view(shape)


class aten_expand_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(-1)
        return x.expand(shape)


class aten_repeat_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        return x.repeat(shape)


class aten_zeros_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        # Keep x in the graph so the output is not a foldable constant.
        return torch.zeros(shape) + x.sum()


class aten_view_size_append_symint(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(y.shape[0])
        return x.view(shape)


class aten_view_size_add_list(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        return x.view(shape + [2])


class aten_view_all_symint(torch.nn.Module):
    def forward(self, x, y):
        return x.view([y.shape[0], y.shape[1], 2])


class aten_permute_append(torch.nn.Module):
    def forward(self, x, y):
        order: List[int] = [1, 0]
        order.append(2)
        return x.permute(order)


# case -> (model, expected op kind, x shape, y shape)
OPS = {
    "view": (aten_view_size_append, "aten::view", [12], [2, 3]),
    "expand": (aten_expand_size_append, "aten::expand", [1, 1, 4], [2, 3]),
    "repeat": (aten_repeat_size_append, "aten::repeat", [2, 3], [2, 3]),
    "zeros": (aten_zeros_size_append, "aten::zeros", [3, 4], [2, 3]),
    "view_append_symint": (aten_view_size_append_symint, "aten::view", [12], [2, 3]),
    "view_add_list": (aten_view_size_add_list, "aten::view", [12], [2, 3]),
    "view_all_symint": (aten_view_all_symint, "aten::view", [12], [2, 3]),
    "permute": (aten_permute_append, "aten::permute", [2, 3, 4], [2, 3]),
}


class TestListConstructConcat(PytorchLayerTest):
    def _prepare_input(self, x_shape, y_shape):
        return (self.random.randn(*x_shape), self.random.randn(*y_shape))

    def create_model(self, case):
        model, kind, _, _ = OPS[case]
        return model(), kind

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("case", list(OPS))
    def test_list_construct_concat(self, case, ie_device, precision, ir_version):
        _, _, x_shape, y_shape = OPS[case]
        self._test(*self.create_model(case), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"x_shape": x_shape, "y_shape": y_shape})
