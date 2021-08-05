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
    axis = np.where(mask == 1)[0]
    new_node = create_op_node_with_second_input(node.graph, op, int64_array(axis))
    node.in_port(0).get_connection().set_destination(new_node.in_port(0))
    node.out_port(0).get_connection().set_source(new_node.out_port(0))

    rename_nodes([(node, node_name + '/ShouldBeDeleted'), (new_node, node_name)])


class ReplaceStridedSliceWithSqueezeUnsqueeze(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].disable_nhwc_to_nchw]

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

            is_shrink_axis_mask = any(x == 1 for x in shrink_axis_mask)
            is_new_axis_mask = any(x == 1 for x in new_axis_mask)

            if is_shrink_axis_mask and is_new_axis_mask:
                unsqueeze_axes = np.where(new_axis_mask == 1)[0]
                squeeze_axes = np.where(shrink_axis_mask == 1)[0]
                assert np.all(unsqueeze_axes != squeeze_axes), 'new_axis_mask and shrink_axis_mask are' \
                                                               'inconsistent: {} and {}'.format(new_axis_mask,
                                                                                                shrink_axis_mask)

                for sq_axis_index, sq_axis in enumerate(squeeze_axes):
                    for unsq_axis_index, unsq_axis in enumerate(unsqueeze_axes):
                        if sq_axis < unsq_axis:
                            unsqueeze_axes[unsq_axis_index] -= 1

                node_name = node.soft_get('name', node.id)
                squeeze_node = create_op_node_with_second_input(graph, Squeeze, squeeze_axes,
                                                                op_attrs=dict(name=node_name + 'Squeeze'))
                unsqueeze_node = create_op_node_with_second_input(graph, Unsqueeze, unsqueeze_axes,
                                                                  input_node=squeeze_node)
                node.in_port(0).get_connection().set_destination(squeeze_node.in_port(0))
                node.out_port(0).get_connection().set_source(unsqueeze_node.out_port(0))

                rename_nodes([(node, node_name + '/ShouldBeDeleted'), (unsqueeze_node, node_name)])

            elif is_shrink_axis_mask and not is_new_axis_mask:
                replace_strided_slice(node, shrink_axis_mask, Squeeze)
            elif not is_shrink_axis_mask and is_new_axis_mask:
                replace_strided_slice(node, new_axis_mask, Unsqueeze)
            else:
                return
