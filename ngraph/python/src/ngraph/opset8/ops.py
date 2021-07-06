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
def multiclass_nms(
    boxes: NodeInput,
    scores: NodeInput,
    sort_result_type: str = "none",
    sort_result_across_batch: bool = False,
    output_type: str = "i64",
    iou_threshold: float = 0.0,
    score_threshold: float = 0.0,
    nms_top_k: int = -1,
    keep_top_k: int = -1,
    background_class: int = -1,
    nms_eta: float = 1.0,
    normalized: bool = True
) -> Node:
    """Return a node which performs MulticlassNms.

    @param boxes: Tensor with box coordinates.
    @param scores: Tensor with box scores.
    @param sort_result_type: Specifies order of output elements
    @param sort_result_across_batch: Specifies whenever it is necessary to sort selected boxes
                                     across batches or not
    @param output_type: Specifies the output tensor type
    @param iou_threshold: Specifies intersection over union threshold
    @param score_threshold: Specifies minimum score to consider box for the processing
    @param nms_top_k: Specifies maximum number of boxes to be selected per class, -1 meaning
                      to keep all boxes
    @param keep_top_k: Specifies maximum number of boxes to be selected per batch element, -1
                       meaning to keep all boxes
    @param background_class: Specifies the background class id, -1 meaning to keep all classes
    @param nms_eta: Specifies eta parameter for adpative NMS, in close range [0, 1.0]
    @param normalized: Specifies whether boxes are normalized or not
    @return: The new node which performs MuticlassNms
    """
    inputs = as_nodes(boxes, scores)

    attributes = {
        "sort_result_type": sort_result_type,
        "sort_result_across_batch": sort_result_across_batch,
        "output_type": output_type,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "nms_top_k": nms_top_k,
        "keep_top_k": keep_top_k,
        "background_class": background_class,
        "nms_eta": nms_eta,
        "normalized": normalized
    }

    return _get_node_factory_opset8().create("MuticlassNms", inputs, attributes)

@nameable_op
def matrix_nms(
    boxes: NodeInput,
    scores: NodeInput,
    sort_result_type: str = "none",
    sort_result_across_batch: bool = False,
    output_type: str = "i64",
    score_threshold: float = 0.0,
    nms_top_k: int = -1,
    keep_top_k: int = -1,
    background_class: int = -1,
    decay_function: str = "linear",
    gaussian_sigma: float = 2.0,
    post_threshold: float = 0.0,
    normalized: bool = True
) -> Node:
    """Return a node which performs MatrixNms.

    @param boxes: Tensor with box coordinates.
    @param scores: Tensor with box scores.
    @param sort_result_type: Specifies order of output elements
    @param sort_result_across_batch: Specifies whenever it is necessary to sort selected boxes
                                     across batches or not
    @param output_type: Specifies the output tensor type
    @param score_threshold: Specifies minimum score to consider box for the processing
    @param nms_top_k: Specifies maximum number of boxes to be selected per class, -1 meaning
                      to keep all boxes
    @param keep_top_k: Specifies maximum number of boxes to be selected per batch element, -1
                       meaning to keep all boxes
    @param background_class: Specifies the background class id, -1 meaning to keep all classes
    @param decay_function: Specifies decay function used to decay scores
    @param gaussian_sigma: Specifies gaussian_sigma parameter for gaussian decay_function
    @param post_threshold: Specifies threshold to filter out boxes with low confidence score 
                           after decaying
    @param normalized: Specifies whether boxes are normalized or not
    @return: The new node which performs MatrixNms
    """
    inputs = as_nodes(boxes, scores)

    attributes = {
        "sort_result_type": sort_result_type,
        "sort_result_across_batch": sort_result_across_batch,
        "output_type": output_type,
        "score_threshold": score_threshold,
        "nms_top_k": nms_top_k,
        "keep_top_k": keep_top_k,
        "background_class": background_class,
        "decay_function": decay_function,
        "gaussian_sigma": gaussian_sigma,
        "post_threshold": post_threshold,
        "normalized": normalized
    }

    return _get_node_factory_opset8().create("MatrixNms", inputs, attributes)
