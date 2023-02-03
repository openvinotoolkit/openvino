# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List, Optional

import ngraph as ng
from ngraph.impl import AxisSet, Node
from ngraph.utils.types import NodeInput, TensorShape, get_dtype, make_constant_node

log = logging.getLogger(__name__)


def get_broadcast_axes(output_shape: TensorShape, input_shape: TensorShape, axis: Optional[int] = None) -> AxisSet:
    """Generate a list of broadcast axes for ngraph++ broadcast.

    Informally, a broadcast "adds" axes to the input tensor,
    replicating elements from the input tensor as needed to fill the new dimensions.
    Function calculate which of the output axes are added in this way.

    :param output_shape: The new shape for the output tensor.
    :param input_shape: The shape of input tensor.
    :param axis: The axis along which we want to replicate elements.
    :return: The indices of added axes.
    """
    axes_indexes = list(range(0, len(output_shape)))
    if axis is None:
        output_begin = len(output_shape) - len(input_shape)
    else:
        output_begin = axis
    right_axes_indexes = list(range(output_begin, output_begin + len(input_shape)))
    for index in reversed(right_axes_indexes):
        del axes_indexes[index]
    return AxisSet(set(axes_indexes))
