# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose
from extensions.middle.StridedSliceNormalizer import StridedSliceNormalizer
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


def replace_strided_slice(node, mask, op):
    node_name = node.soft_get('name', node.id)
    axis = np.where(mask == 1)
    new_node = create_op_node_with_second_input(node.graph, op, int64_array(axis))
    node.in_port(0).get_connection().set_destination(new_node.in_port(0))
    node.out_port(0).get_connection().set_source(new_node.out_port(0))

    rename_nodes([(node, node_name + '/ShouldBeDeleted'), (new_node, node_name)])


class ReplaceStridedSliceWithSqueezeUnsqueeze(MiddleReplacementPattern):

    enabled = True
    force_clean_up = True

    def run_before(self):
        return [InsertLayoutPropagationTranspose]

    def run_after(self):
        return [StridedSliceNormalizer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='StridedSlice'):
            input_shape = node.in_port(0).data.get_shape()
            output_shape = node.out_port(0).data.get_shape()

            if np.prod(input_shape) != np.prod(output_shape):
                return

            shrink_axis_mask = node.soft_get('shrink_axis_mask', np.zeros(len(input_shape)))
            new_axis_mask = node.soft_get('new_axis_mask', np.zeros(len(input_shape)))

            if all(x == 0 for x in shrink_axis_mask):
                if all(y == 0 for y in new_axis_mask):
                    return
                else:
                    replace_strided_slice(node, new_axis_mask, Unsqueeze)
            else:
                if any(y != 0 for y in new_axis_mask):
                    return
                else:
                    replace_strided_slice(node, shrink_axis_mask, Squeeze)
