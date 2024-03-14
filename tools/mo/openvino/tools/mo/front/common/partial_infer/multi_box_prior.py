# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Node


def multi_box_prior_infer_mxnet(node: Node):
    v10 = node.has_and_set('V10_infer')
    data_H, data_W = node.in_node(0).value if v10 else node.in_node(0).shape[2:]

    num_ratios = len(node.aspect_ratio)
    num_priors = len(node.min_size) + num_ratios - 1
    if v10:
        node.out_node(0).shape = shape_array([2, data_H * data_W * num_priors * 4])
    else:
        node.out_node(0).shape = shape_array([1, 2, data_H * data_W * num_priors * 4])
