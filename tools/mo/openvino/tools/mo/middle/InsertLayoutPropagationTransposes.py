# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.middle.pass_separator import PostMiddleStart
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph, Node, Port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.op import PermuteAttrs


class InsertLayoutPropagationTranspose(MiddleReplacementPattern):
    """
    The transformation inserts Transpose layers before/after operations that change the interpretation of data, for
    example, Reshape from 3D to 4D or from 4D to 3D. These Transpose layers basically convert layout from N(D)HWC to
    NC(D)HW and in the reverse order.
    """
    enabled = True
    force_clean_up = True  # need to run clean up after the transformation to update shapes
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def run_after(self):
        return [PostMiddleStart]

    def run_before(self):
        return []

    @staticmethod
    def is_nchw_to_nhwc_transpose_needed(node: Node):
        """
        The function checks that it is necessary to insert Transpose from NCHW to NHWC before the node.
        The transpose is needed when all the following conditions are met:
         1. The node is marked as 'reinterp_shape' attribute
         2. The node is *not* marked as getting input in correct layout (implicitly imply that the input is on port 0)
         3. The input shape rank is not less than 4
         4. Node is not a part of shape sub-graph (layout permutation is handled separately for such a sub-graph)

        :param node: node to check
        :return: result of the check
        """
        return node.has_and_set('reinterp_shape') and \
               not is_input_data_in_correct_layout(node, 0) and \
               len(node.in_port(0).data.get_shape()) >= 4 and \
               all([port.data.get_value() is None for port in node.out_ports().values() if not port.disconnected()])

    @staticmethod
    def is_nhwc_to_nchw_transpose_needed(node: Node):
        """
        The function checks that it is necessary to insert Transpose from NHWC to NCHW after the node.
        The transpose is needed when all the following conditions are met:
         1. The node is marked as 'reinterp_shape' attribute
         2. The node is *not* marked as generating output in correct layout (implicitly imply that the output port is 0)
         3. The output shape rank is not less than 4
         4. Node is not a part of shape sub-graph (layout permutation is handled separately for such a sub-graph)
        :param node: node to check
        :return: result of the check
        """
        return node.has_and_set('reinterp_shape') and \
               not is_output_data_in_correct_layout(node, 0) and \
               len(node.out_port(0).data.get_shape()) >= 4 and \
               all([port.data.get_value() is None for port in node.out_ports().values() if not port.disconnected()])

    def find_and_replace_pattern(self, graph: Graph):

        # we need to import these functions here to avoid circular dependent imports
        from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input

        if graph.graph['layout'] != 'NHWC':
            # we check it here because this transformation is called explicitly from the pipeline
            return

        # reshape from 4D-5D -> ND. Insert Transpose(NC(D)HW->N(D)HWC) before Reshape
        for reinterp_shape_node_id in graph.get_nodes_with_attributes(reinterp_shape=True):
            reinterp_shape_node = Node(graph, reinterp_shape_node_id)
            assert 0 in reinterp_shape_node.in_nodes(), 'Node {} does not have 0 input. \n{}'.format(
                reinterp_shape_node_id, graph.dump_graph_for_graphviz())
            input_shape = reinterp_shape_node.in_node(0).shape
            if self.is_nchw_to_nhwc_transpose_needed(reinterp_shape_node):
                permute_node = create_op_node_with_second_input(
                    graph, Transpose, PermuteAttrs().get_nchw_to_nhwc_permutation(len(input_shape)).perm,
                    {'name': reinterp_shape_node.in_port(0).get_source().node.name + '/Transpose'}
                )
                reinterp_shape_node.in_port(0).get_connection().insert_node(permute_node)

                order_const = permute_node.in_port(1).get_source().node
                order_const.infer(order_const)
                # do not infer the Transpose node because it should have input data node in NCHW layout (but currently
                # it is NHWC because data node attributes has not been permuted yet) and produce output in NHWC layout
                # (which is true at this moment)
                permute_node['need_shape_inference'] = False
                # mark the Transpose output data node having correct layout so it's shape will not be permuted
                mark_output_as_in_correct_layout(permute_node, 0)

                # keep the reinterp_shape_node in NHWC layout
                for in_port_id, _ in reinterp_shape_node.in_ports().items():
                    mark_input_as_in_correct_layout(reinterp_shape_node, in_port_id)

        # reshape from ND -> 4D-5D. Insert Transpose(N(D)HWC->NC(D)HW) after Reshape
        for reinterp_shape_node_id in graph.get_nodes_with_attributes(reinterp_shape=True):
            reinterp_shape_node = Node(graph, reinterp_shape_node_id)
            assert 0 in reinterp_shape_node.out_nodes(), 'Node {} does not have 0 output. \n{}'.format(
                reinterp_shape_node_id, graph.dump_graph_for_graphviz())
            output_shape = reinterp_shape_node.out_node(0).shape
            if self.is_nhwc_to_nchw_transpose_needed(reinterp_shape_node):
                permute_node = create_op_node_with_second_input(
                    graph, Transpose, PermuteAttrs().get_nhwc_to_nchw_permutation(len(output_shape)).perm,
                    {'name': reinterp_shape_node.id + '/Transpose'})
                reinterp_shape_node.out_port(0).get_connection().insert_node(permute_node)

                # the Reshape and Transpose operations should work in original (NHWC layout) so the Transpose
                # will convert it to the NCHW
                mark_input_as_in_correct_layout(permute_node, 0)
                mark_input_as_in_correct_layout(permute_node, 1)
                # do not set Transpose output data node 'correct_data_layout' attribute so the data node shape will be
                # permuted

                # keep the reinterp_shape_node in NHWC layout
                mark_output_as_in_correct_layout(reinterp_shape_node, 0)
                for in_port_id in reinterp_shape_node.in_ports().keys():
                    if in_port_id:
                        mark_input_as_in_correct_layout(reinterp_shape_node, in_port_id)

                # do not re-infer the Transpose node because it output data node should be in NHWC layout to make the
                # rest of the graph consistent
                permute_node['need_shape_inference'] = False


def is_input_data_in_correct_layout(node: Node, port_ind: int):
    assert node.soft_get('kind') == 'op', 'The function work with operation nodes only'
    return 'correct_in_data_layout' in node.attrs() and port_ind in node.attrs()['correct_in_data_layout']


def mark_input_as_in_correct_layout(node: Node, port_ind: int):
    assert node.soft_get('kind') == 'op', 'The function work with operation nodes only'
    graph = node.graph
    graph.node[node.id].setdefault('correct_in_data_layout', set())
    graph.node[node.id]['correct_in_data_layout'].add(port_ind)


def is_output_data_in_correct_layout(node: Node, port_ind: int):
    assert node.soft_get('kind') == 'op', 'The function work with operation nodes only'
    return 'correct_out_data_layout' in node.attrs() and port_ind in node.attrs()['correct_out_data_layout']


def mark_output_as_in_correct_layout(node: Node, port_ind: int):
    assert node.soft_get('kind') == 'op', 'The function work with operation nodes only'
    graph = node.graph
    graph.node[node.id].setdefault('correct_out_data_layout', set())
    graph.node[node.id]['correct_out_data_layout'].add(port_ind)


def mark_as_correct_data_layout(node: Node):
    """
    The analogue of the attribute 'correct_data_layout' for the operation node
    :param node: node to mark it with attribute 'correct_data_layout'
    :return: None
    """
    assert node.soft_get('kind') == 'op', 'The function work with operation nodes only'
    for ind, port in node.in_ports().items():
        mark_input_as_in_correct_layout(node, ind)

    for ind, port in node.out_ports().items():
        mark_output_as_in_correct_layout(node, ind)


def insert_transpose(graph: Graph, input_port: Port, before_input=True):
    from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs

    input_rank = len(input_port.data.get_shape())
    if input_rank > 3:
        if before_input:
            axis_order = np.concatenate((int64_array([0]),
                                         int64_array(list(range(2, input_rank))),
                                         int64_array([1])))
            source_node = input_port.get_source().node
            transpose_name = source_node.soft_get('name', source_node.id) + '/TransposeToNHWC'
        else:
            axis_order = np.concatenate(
                (int64_array([0]),
                 int64_array([input_rank - 1]),
                 int64_array(list(range(1, input_rank - 1)))))
            transpose_name = input_port.node.soft_get('name', input_port.node.id) + '/TransposeToNCHW'
            input_port.node['need_shape_inference'] = True
            input_port.node['override_output_shape'] = True
        transpose = create_op_with_const_inputs(graph, Transpose, {1: axis_order}, {'name': transpose_name})
        input_port.get_connection().insert_node(transpose)
        transpose['need_shape_inference'] = True
        transpose['override_output_shape'] = True
