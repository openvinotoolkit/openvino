# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose
from openvino.tools.mo.middle.StridedSliceNormalizer import StridedSliceNormalizer
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class ReplaceStridedSliceShrinkNewAxisWithSqueezeUnsqueeze(MiddleReplacementPattern):
    r"""
    The transformation removes shrink_axis and new_axis masks from StridedSlice and inserts Squeeze/Unsqueeze nodes
    instead of these masks.
    This is necessary if StridedSlice is to be executed in original N(D)HWC layout, because
    the operation does not have reinterp_shape attribute and MO can not insert NC(D)HW -> N(D)HWC Transpose in
    openvino/tools/mo/middle/InsertLayoutPropagationTransposes.py.
    """
    enabled = True

    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def run_before(self):
        return [InsertLayoutPropagationTranspose]

    def run_after(self):
        return [StridedSliceNormalizer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='StridedSlice'):
            input_shape = node.in_port(0).data.get_shape()

            shrink_axis_mask = node.soft_get('shrink_axis_mask', np.zeros(len(input_shape)))
            new_axis_mask = node.soft_get('new_axis_mask', np.zeros(len(input_shape)))

            squeeze_indices = np.nonzero(shrink_axis_mask)[0]
            unsqueeze_indices = np.nonzero(new_axis_mask)[0]
            for i, axis in enumerate(unsqueeze_indices):
                unsqueeze_indices[i] -= np.count_nonzero(squeeze_indices < unsqueeze_indices[i])

            latest_node = node
            if len(squeeze_indices) > 0:
                squeeze_node = create_op_node_with_second_input(node.graph, Squeeze, int64_array(squeeze_indices))
                node.out_port(0).get_connection().insert_node(squeeze_node)
                node.shrink_axis_mask = np.zeros(len(input_shape))
                latest_node = squeeze_node
                node['need_shape_inference'] = True
                node['override_output_shape'] = True
            if len(unsqueeze_indices) > 0:
                unsqueeze_node = create_op_node_with_second_input(node.graph, Unsqueeze, int64_array(unsqueeze_indices))
                latest_node.out_port(0).get_connection().insert_node(unsqueeze_node)
                node.new_axis_mask = np.zeros(len(input_shape))
                node['need_shape_inference'] = True
                node['override_output_shape'] = True
