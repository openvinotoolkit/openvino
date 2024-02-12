# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset13."""
from functools import partial
from typing import Optional, List
import logging

log = logging.getLogger(__name__)

from openvino.runtime import Node
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import nameable_op
from openvino.runtime.utils.types import (
    NodeInput,
    as_node,
    TensorShape,
)

_get_node_factory_opset14 = partial(_get_node_factory, "opset14")


# -------------------------------------------- ops ------------------------------------------------

@nameable_op
def max_pool(
    data: NodeInput,
    strides: List[int],
    dilations: List[int],
    pads_begin: List[int],
    pads_end: List[int],
    kernel_shape: TensorShape,
    rounding_type: str = "floor",
    auto_pad: Optional[str] = None,
    index_element_type: Optional[str] = "i64",
    axis: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Perform max pooling operation and return both values and indices of the selected elements.

    :param  data:                The node providing input data.
    :param  strides:             The distance (in pixels) to slide the filter on the feature map
                                 over the axes.
    :param  dilations:           The dilation of filter elements(distance between elements).
    :param  pads_begin:          The number of pixels to add at the beginning along each axis.
    :param  pads_end:            The number of pixels to add at the end along each axis.
    :param  kernel_shape:        The pooling operation kernel shape.
    :param  rounding_type:       Determines used rounding schema when computing output shape.
                                 Acceptable values are: ['floor', 'ceil']. Defaults to 'floor'.
    :param  auto_pad:            Determines how the padding is calculated. Acceptable values:
                                 [None, 'same_upper', 'same_lower', 'valid']. Defaults to None.
    :param  index_element_type:  The data type used for the indices output of this operator.
                                 Defaults to i64.
    :param  axis:                The first dimension in the data shape used to determine the maximum
                                 returned index value. The value is the product of all dimensions
                                 starting at the provided axis. Defaults to 0.
    :param  name:                The optional name for the created output node.

    :return:   The new node performing max pooling operation.
    """
    if auto_pad is None:
        auto_pad = "explicit"
    return _get_node_factory_opset14().create(
        "MaxPool",
        [as_node(data)],
        {
            "strides": strides,
            "dilations": dilations,
            "pads_begin": pads_begin,
            "pads_end": pads_end,
            "kernel": kernel_shape,
            "rounding_type": rounding_type.upper(),
            "auto_pad": auto_pad.upper(),
            "index_element_type": index_element_type,
            "axis": axis,
        },
    )
