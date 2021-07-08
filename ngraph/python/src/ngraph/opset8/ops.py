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
def gather(
        data: NodeInput,
        indices: NodeInput,
        axis: NodeInput,
        batch_dims: Optional[int] = 0,
) -> Node:
    """Return a node which performs Gather with support of negative indices.

    @param data:         N-D tensor with data for gathering
    @param indices:      N-D tensor with indices by which data is gathered
    @param axis:         axis along which elements are gathered
    @param batch_dims:   number of batch dimensions
    @return:             The new node which performs Gather
    """
    inputs = as_nodes(data, indices, axis)
    attributes = {
        "batch_dims": batch_dims
    }
    return _get_node_factory_opset8().create("Gather", inputs, attributes)
