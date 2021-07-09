# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all ngraph ops."""
from functools import partial
from typing import Callable, Iterable, List, Optional, Set, Union

import numpy as np
from ngraph.impl import Node, Shape
from ngraph.impl.op import Constant, Parameter
from ngraph.opset_utils import _get_node_factory
from ngraph.utils.decorators import binary_op, nameable_op, unary_op
from ngraph.utils.input_validation import (
    assert_list_of_ints,
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from ngraph.utils.node_factory import NodeFactory
from ngraph.utils.tensor_iterator_types import (
    GraphBody,
    TensorIteratorSliceInputDesc,
    TensorIteratorMergedInputDesc,
    TensorIteratorInvariantInputDesc,
    TensorIteratorBodyOutputDesc,
    TensorIteratorConcatOutputDesc,
)
from ngraph.utils.types import (
    NodeInput,
    NumericData,
    NumericType,
    ScalarData,
    TensorShape,
    as_node,
    as_nodes,
    get_dtype,
    get_element_type,
    get_element_type_str,
    make_constant_node,
)

_get_node_factory_opset8 = partial(_get_node_factory, "opset8")


# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def deformable_convolution(
        data: NodeInput,
        offsets: NodeInput,
        filters: NodeInput,
        strides: List[int],
        pads_begin: List[int],
        pads_end: List[int],
        dilations: List[int],
        mask: Optional[NodeInput] = None,
        auto_pad: str = "EXPLICIT",
        group: int = 1,
        deformable_group: int = 1,
        bilinear_interpolation_pad: bool = False,
        name: Optional[str] = None,
) -> Node:
    """Return a node which performs deformable convolution operation.

    @param data: The node providing data batch tensor.
    @param offsets: The node providing offset tensor.
    @param filters: The node providing filters tensor.
    @param strides: The distance (in pixels) to slide the filter on the feature map over the axes.
    @param pads_begin: The number of pixels to add to the beginning along each axis.
    @param pads_end: The number of pixels to add to the end along each axis.
    @param dilations: The distance in width and height between elements (weights) in the filter.
    @param mask: The node providing modulation scalar (mask) tensor.
    @param auto_pad: The type of padding. Range of values: explicit, same_upper, same_lower, valid.
    @param group: The number of groups which both output and input should be split into.
    @param deformable_group: The number of groups which deformable values and output should be split
                             into along the channel axis.
    @param bilinear_interpolation_pad: The flag that determines the mode of bilinear interpolation
                                               execution.
    @param name: The optional new name for output node.
    @return New node performing deformable convolution operation.
    """
    if mask is None:
        inputs = as_nodes(data, offsets, filters)
    else:
        inputs = as_nodes(data, offsets, filters, mask)

    return _get_node_factory_opset8().create(
        "DeformableConvolution",
        inputs,
        {
            "strides": strides,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "dilations": dilations,
            "auto_pad": auto_pad,
            "group": group,
            "deformable_group": deformable_group,
            "bilinear_interpolation_pad": bilinear_interpolation_pad
        },
    )
