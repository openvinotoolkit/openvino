# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose
from openvino.tools.mo.middle.StridedSliceNormalizer import StridedSliceNormalizer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, rename_nodes, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


def replace_strided_slice(node: Node, mask: np.ndarray, op: callable):
    node_name = node.soft_get('name', node.id)
    axes = np.where(mask == 1)[0]
    new_node = create_op_node_with_second_input(node.graph, op, int64_array(axes))
    node.in_port(0).get_connection().set_destination(new_node.in_port(0))
    node.out_port(0).get_connection().set_source(new_node.out_port(0))

    rename_nodes([(node, node_name + '/ShouldBeDeleted'), (new_node, node_name)])
    node.graph.remove_node(node.id)


class ReplaceStridedSliceWithSqueezeUnsqueeze(MiddleReplacementPattern):
    r"""
    The transformation replaces StridedSlice with a Squeeze/Unsqueeze node if StridedSlice executes like a Squeeze/Unsqueeze
    and does not slice values. This is necessary if StridedSlice is to be executed in original N(D)HWC layout, because
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
            output_shape = node.out_port(0).data.get_shape()

            if np.prod(input_shape) != np.prod(output_shape):
                continue

            shrink_axis_mask = node.soft_get('shrink_axis_mask', np.zeros(len(input_shape), dtype=np.bool))
            new_axis_mask = node.soft_get('new_axis_mask', np.zeros(len(input_shape), dtype=np.bool))

            is_shrink_axis_mask = any(x == 1 for x in shrink_axis_mask)
            is_new_axis_mask = any(x == 1 for x in new_axis_mask)

            if is_shrink_axis_mask and is_new_axis_mask:
                # TODO: make it in a separate ticket
                continue
            elif is_shrink_axis_mask and not is_new_axis_mask:
                replace_strided_slice(node, shrink_axis_mask, Squeeze)
            elif not is_shrink_axis_mask and is_new_axis_mask:
                replace_strided_slice(node, new_axis_mask, Unsqueeze)
