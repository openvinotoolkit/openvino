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
from functools import partial

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

_get_node_factory_opset4 = partial(_get_node_factory, "opset4")

# -------------------------------------------- ops ------------------------------------------------


@nameable_op
def ctc_loss(
    logits: NodeInput,
    logit_length: NodeInput,
    labels: NodeInput,
    label_length: NodeInput,
    blank_index: Optional[NodeInput] = None,
    preprocess_collapse_repeated: bool = False,
    ctc_merge_repeated: bool = True,
    unique: bool = False,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs CTCLoss.

    :param logits:                        3-D tensor of logits.
    :param logit_length:                  1-D tensor of lengths for each object from a batch.
    :param labels:                        2-D tensor of labels for which likelihood is estimated using logits.
    :param label_length:                  1-D tensor of length for each label sequence.
    :param blank_index:                   Scalar used to mark a blank index.
    :param preprocess_collapse_repeated:  Flag for preprocessing labels before loss calculation.
    :param ctc_merge_repeated:            Flag for merging repeated characters in a potential alignment.
    :param unique:                        Flag to find unique elements in a target.
    :return: The new node which performs CTCLoss
    """
    if blank_index is not None:
        inputs = as_nodes(logits, logit_length, labels, label_length, blank_index)
    else:
        inputs = as_nodes(logits, logit_length, labels, label_length)

    attributes = {
        "preprocess_collapse_repeated": preprocess_collapse_repeated,
        "ctc_merge_repeated": ctc_merge_repeated,
        "unique": unique,
    }

    return _get_node_factory_opset4().create("CTCLoss", inputs, attributes)


@nameable_op
def non_max_suppression(
    boxes: NodeInput,
    scores: NodeInput,
    max_output_boxes_per_class: Optional[NodeInput] = None,
    iou_threshold: Optional[NodeInput] = None,
    score_threshold: Optional[NodeInput] = None,
    box_encoding: str = "corner",
    sort_result_descending: bool = True,
    output_type: str = "i64",
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs NonMaxSuppression.

    :param boxes: Tensor with box coordinates.
    :param scores: Tensor with box scores.
    :param max_output_boxes_per_class: Tensor Specifying maximum number of boxes
                                        to be selected per class.
    :param iou_threshold: Tensor specifying intersection over union threshold
    :param score_threshold: Tensor specifying minimum score to consider box for the processing.
    :param box_encoding: Format of boxes data encoding.
    :param sort_result_descending: Flag that specifies whenever it is necessary to sort selected
                                   boxes across batches or not.
    :param output_type: Output element type.
    :return: The new node which performs NonMaxSuppression
    """
    if max_output_boxes_per_class is None:
        max_output_boxes_per_class = make_constant_node(0, np.int64)
    if iou_threshold is None:
        iou_threshold = make_constant_node(0, np.float32)
    if score_threshold is None:
        score_threshold = make_constant_node(0, np.float32)

    inputs = as_nodes(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    attributes = {
        "box_encoding": box_encoding,
        "sort_result_descending": sort_result_descending,
        "output_type": output_type,
    }

    return _get_node_factory_opset4().create("NonMaxSuppression", inputs, attributes)


@nameable_op
def mish(data: NodeInput, name: Optional[str] = None,) -> Node:
    """Return a node which performs Mish.

    :param data: Tensor with input data floating point type.
    :return: The new node which performs Mish
    """
    return _get_node_factory_opset4().create("Mish", as_nodes(data), {})
