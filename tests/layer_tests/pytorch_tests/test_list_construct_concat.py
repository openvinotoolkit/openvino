# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Coverage for flatten_list_element_for_concat() in src/frontends/pytorch/src/utils.cpp.

A shape-building list can mix elements of different ranks: `aten::size` with no
dim is a whole ShapeOf and is rank 1, while a literal int or `aten::size(x, dim)`
is rank 0. Every element has to be made 1-D before Concat joins them along axis 0,
and doing that with Unsqueeze(0) turns the rank-1 element into [1, N] so Concat
rejects the mixed ranks:

    Check 'TRShape::merge_into(output_shape, in_copy)' failed at
    concat_shape_inference.hpp:43 ... Shape inference input shapes {[1,?],[1]}

The models below build such a list (`y.size()` followed by `append`) and feed it
to the shape-list consumers that route through concat_list_construct() /
get_list_as_outputs(unsqueeze_for_concat=true). The list stays symbolic because
the layer test harness reshapes every input to fully dynamic.
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


class aten_reshape_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        return x.reshape(shape)


class aten_expand_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(-1)
        return x.expand(shape)


class aten_broadcast_to_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(4)
        return torch.broadcast_to(x, shape)


class aten_repeat_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        return x.repeat(shape)


class aten_new_zeros_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        return x.new_zeros(shape)


class aten_zeros_size_append(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        # Keep x in the graph so the output is not a foldable constant.
        return torch.zeros(shape) + x.sum()


class aten_view_size_append_twice(torch.nn.Module):
    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(2)
        shape.append(3)
        return x.view(shape)


class aten_view_size_append_symint(torch.nn.Module):
    """The appended element is itself symbolic (rank 0), not a literal."""

    def forward(self, x, y):
        shape: List[int] = y.size()
        shape.append(y.shape[0])
        return x.view(shape)


class aten_view_size_add_list(torch.nn.Module):
    """List concatenation instead of append: goes through the aten::add branch."""

    def forward(self, x, y):
        shape: List[int] = y.size()
        return x.view(shape + [2])


class aten_view_all_symint(torch.nn.Module):
    """Every element is rank 0 - the uniform-rank baseline."""

    def forward(self, x, y):
        return x.view([y.shape[0], y.shape[1], 2])


class aten_permute_append(torch.nn.Module):
    def forward(self, x, y):
        order: List[int] = [1, 0]
        order.append(2)
        return x.permute(order)


class aten_mean_append(torch.nn.Module):
    def forward(self, x, y):
        dims: List[int] = [0]
        dims.append(1)
        return x.mean(dims)


# case -> (model, expected op kind, x shape, y shape)
OPS = {
    "view": (aten_view_size_append, "aten::view", [12], [2, 3]),
    "reshape": (aten_reshape_size_append, "aten::reshape", [12], [2, 3]),
    "expand": (aten_expand_size_append, "aten::expand", [1, 1, 4], [2, 3]),
    "broadcast_to": (aten_broadcast_to_size_append, "aten::broadcast_to", [1, 1, 4], [2, 3]),
    "repeat": (aten_repeat_size_append, "aten::repeat", [2, 3], [2, 3]),
    "new_zeros": (aten_new_zeros_size_append, "aten::new_zeros", [3, 4], [2, 3]),
    "zeros": (aten_zeros_size_append, "aten::zeros", [3, 4], [2, 3]),
    "view_two_appends": (aten_view_size_append_twice, "aten::view", [36], [2, 3]),
    "view_append_symint": (aten_view_size_append_symint, "aten::view", [12], [2, 3]),
    "view_add_list": (aten_view_size_add_list, "aten::view", [12], [2, 3]),
    "view_all_symint": (aten_view_all_symint, "aten::view", [12], [2, 3]),
    "permute": (aten_permute_append, "aten::permute", [2, 3, 4], [2, 3]),
    "mean": (aten_mean_append, "aten::mean", [2, 3, 4], [2, 3]),
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
