# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

"""Factory functions for all ngraph ops."""
from typing import Callable, Iterable, List, Optional, Set, Union

import numpy as np

from ngraph.impl import Node, Shape
from ngraph.impl.op import Constant, GetOutputElement, Parameter
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


def _get_node_factory(opset_version: Optional[str] = "opset2") -> NodeFactory:
    """Return NodeFactory configured to create operators from specified opset version."""
    if opset_version:
        return NodeFactory(opset_version)
    else:
        return NodeFactory()


# ------------------------ ops ---------------------------------------------------

@nameable_op
def batch_to_space(
    data: NodeInput,
    block_shape: NodeInput,
    crops_begin: NodeInput,
    crops_end: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform BatchToSpace operation on the input tensor.

    BatchToSpace permutes data from the batch dimension of the data tensor into spatial dimensions.

    :param data: Node producing the data tensor.
    :param block_shape: The sizes of the block of values to be moved.
    :param crops_begin: Specifies the amount to crop from the beginning along each axis of `data`.
    :param crops_end: Specifies the amount to crop from the end along each axis of `data`.
    :param name: Optional output node name.
    :return: The new node performing a BatchToSpace operation.
    """
    return _get_node_factory().create(
        "BatchToSpace", as_nodes(data, block_shape, crops_begin, crops_end)
    )

@nameable_op
def mvn(
    data: Node,
    across_channels: bool = False,
    normalize_variance: bool = False,
    eps: float = 1e-9,
    name: str = None,
) -> Node:
    r"""Perform Mean Variance Normalization operation on data from input node.

    Computes MVN on the input tensor :code:`data` (called `X`) using formula:

    .. math:: Y = \dfrac{X-EX}{\sqrt{E(X-EX)^2}}

    :param data: The node with data tensor.
    :param across_channels: Denotes if mean values are shared across channels.
    :param normalize_variance: Denotes whether to perform variance normalization.
    :param eps: The number added to the variance to avoid division by zero
               when normalizing the value. Scalar value.
    :param name: Optional output node name.
    :return: The new node performing a MVN operation on input tensor.
    """
    return _get_node_factory().create(
        "MVN",
        [data],
        {"across_channels": across_channels, "normalize_variance": normalize_variance, "eps": eps},
    )


@nameable_op
def reorg_yolo(input: Node, stride: List[int], name: Optional[str] = None) -> Node:
    """Return a node which produces the ReorgYolo operation.

    :param input:   Input data
    :param stride:  Stride to reorganize input by
    :param name:    Optional name for output node.
    :return: ReorgYolo node
    """
    return _get_node_factory().create("ReorgYolo", [input], {"stride": stride})


@nameable_op
def roi_pooling(
    input: NodeInput,
    coords: NodeInput,
    output_size: TensorShape,
    spatial_scale: NumericData,
    method: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which produces an ROIPooling operation.

    :param input:          Input feature map {N, C, ...}
    :param coords:         Coordinates of bounding boxes
    :param output_size:    Height/Width of ROI output features (shape)
    :param spatial_scale:  Ratio of input feature map over input image size (float)
    :param method:         Method of pooling - string: "max" or "bilinear"
    :return:               ROIPooling node
    """
    method = method.lower()
    return _get_node_factory().create(
        "ROIPooling",
        as_nodes(input, coords),
        {"output_size": Shape(output_size), "spatial_scale": spatial_scale, "method": method},
    )


@nameable_op
def space_to_batch(
    data: NodeInput,
    block_shape: NodeInput,
    pads_begin: NodeInput,
    pads_end: NodeInput,
    name: Optional[str] = None,
) -> Node:
    """Perform SpaceToBatch operation on the input tensor.

    SpaceToBatch permutes data tensor blocks of spatial data into batch dimension.
    The operator returns a copy of the input tensor where values from spatial blocks dimensions
    are moved in the batch dimension

    :param data: Node producing the data tensor.
    :param block_shape: The sizes of the block of values to be moved.
    :param pads_begin: Specifies the padding for the beginning along each axis of `data`.
    :param pads_end: Specifies the padding for the ending along each axis of `data`.
    :param name: Optional output node name.
    :return: The new node performing a SpaceToBatch operation.
    """
    return _get_node_factory().create(
        "SpaceToBatch", as_nodes(data, block_shape, pads_begin, pads_end)
    )