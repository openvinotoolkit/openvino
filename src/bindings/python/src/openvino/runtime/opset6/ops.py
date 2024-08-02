# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for all openvino ops."""
from typing import Callable, Iterable, List, Optional, Set, Union, Dict, Any

import numpy as np
from functools import partial, singledispatch

from openvino.runtime import Node, Shape, Type, PartialShape
from openvino.runtime.op import assign, Constant, Parameter
from openvino.runtime.opset_utils import _get_node_factory
from openvino.runtime.utils.decorators import binary_op, nameable_op, unary_op
from openvino.runtime.utils.input_validation import (
    assert_list_of_ints,
    check_valid_attributes,
    is_non_negative_value,
    is_positive_value,
)
from openvino.runtime.utils.node_factory import NodeFactory
from openvino.runtime.utils.types import (
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

_get_node_factory_opset6 = partial(_get_node_factory, "opset6")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def ctc_greedy_decoder_seq_len(
    data: NodeInput,
    sequence_length: NodeInput,
    blank_index: Optional[NodeInput] = None,
    merge_repeated: bool = True,
    classes_index_type: str = "i32",
    sequence_length_type: str = "i32",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs CTCGreedyDecoderSeqLen.

    :param data:            The input 3D tensor. Shape: [batch_size, seq_length, num_classes]
    :param sequence_length: Input 1D tensor with sequence length. Shape: [batch_size]
    :param blank_index:     Scalar or 1D tensor with specifies the class index to use for the blank class.
                            Optional parameter. Default value is num_classes-1.
    :return:                The new node which performs CTCGreedyDecoderSeqLen.
    """
    if blank_index is not None:
        inputs = as_nodes(data, sequence_length, blank_index, name=name)
    else:
        inputs = as_nodes(data, sequence_length, name=name)

    attributes = {
        "merge_repeated": merge_repeated,
        "classes_index_type": classes_index_type,
        "sequence_length_type": sequence_length_type,
    }

    return _get_node_factory_opset6().create("CTCGreedyDecoderSeqLen", inputs, attributes)


@nameable_op
def gather_elements(
    data: NodeInput,
    indices: NodeInput,
    axis: Optional[int] = 0,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs GatherElements.

    :param data:       N-D tensor with data for gathering
    :param indices:    N-D tensor with indices by which data is gathered
    :param axis:       axis along which elements are gathered
    :return:           The new node which performs GatherElements
    """
    inputs = as_nodes(data, indices, name=name)

    attributes = {
        "axis": axis,
    }

    return _get_node_factory_opset6().create("GatherElements", inputs, attributes)


@nameable_op
def mvn(
    data: Node,
    axes: Node,
    normalize_variance: bool,
    eps: float,
    eps_mode: str,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs MeanVarianceNormalization (MVN).

    :param data: The node with data tensor.
    :param axes: The node with axes to reduce on.
    :param normalize_variance: Denotes whether to perform variance normalization.
    :param eps: The number added to the variance to avoid division by zero
               when normalizing the value. Scalar value.
    :param eps_mode: how eps is applied (`inside_sqrt` or `outside_sqrt`)
    :param name: Optional output node name.
    :return: The new node performing a MVN operation on input tensor.
    """
    inputs = as_nodes(data, axes, name=name)

    attributes = {
        "normalize_variance": normalize_variance,
        "eps": eps,
        "eps_mode": eps_mode,
    }

    return _get_node_factory_opset6().create("MVN", inputs, attributes)


@singledispatch
@nameable_op
def read_value(init_value: NodeInput,
               variable_id: str,
               variable_type: Optional[Union[NumericType, Type, str]] = None,
               variable_shape: Optional[TensorShape] = None,
               name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param init_value:   Node producing a value to be returned instead of an unassigned variable.
    :param variable_id:  Id of a variable to be read.
    :param variable_type:   Optional type to be set into Variable.
    :param variable_shape:  Optional shape to be set into Variable.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    attr_map: Dict[str, Any] = {"variable_id": variable_id}

    if variable_type is not None:
        if not isinstance(variable_type, Type) and not isinstance(variable_type, str):
            attr_map["variable_type"] = get_element_type_str(variable_type)
        else:
            attr_map["variable_type"] = variable_type

    if variable_shape is not None:
        attr_map["variable_shape"] = PartialShape(variable_shape)

    return _get_node_factory_opset6().create(
        "ReadValue",
        [as_node(init_value, name=name)],
        attr_map,
    )


@read_value.register
def _(variable_id: str,
      variable_type: Optional[Union[NumericType, Type, str]] = None,
      variable_shape: Optional[TensorShape] = None,
      name: Optional[str] = None) -> Node:
    """Return a node which produces the Assign operation.

    :param variable_id:  Id of a variable to be read.
    :param variable_type:   Optional type to be set into Variable.
    :param variable_shape:  Optional shape to be set into Variable.
    :param name:         Optional name for output node.
    :return: ReadValue node
    """
    attr_map: Dict[str, Any] = {"variable_id": variable_id}

    if variable_type is not None:
        if not isinstance(variable_type, Type) and not isinstance(variable_type, str):
            attr_map["variable_type"] = get_element_type_str(variable_type)
        else:
            attr_map["variable_type"] = variable_type

    if variable_shape is not None:
        attr_map["variable_shape"] = PartialShape(variable_shape)

    return _get_node_factory_opset6().create(
        "ReadValue",
        [],
        attr_map,
    )
