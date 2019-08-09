"""
 Copyright (c) 2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
from extensions.ops.transpose import Transpose
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, Node
from mo.middle.passes.eliminate import graph_clean_up_tf
from mo.ops.const import Const
from mo.ops.op import PermuteAttrs


class InsertLayoutPropagationTranspose(BackReplacementPattern):
    """
    The transformation inserts Transpose layers before/after Reshape layers that change the interpretation of data, for
    example, from 3D to 4D or from 4D to 3D. These Transpose layers basically convert layout from N(D)HWC to NC(D)HW and
    in the reverse order.

    The transformation is explicitly executed in the pipeline.
    """
    enabled = False
    force_clean_up = True  # need to run clean up after the transformation to update shapes
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']

    def find_and_replace_pattern(self, graph: Graph):
        if graph.graph['layout'] != 'NHWC':
            # we check it here because this transformation is called explicitly from the pipeline
            return

        # reshape from 4D-5D -> ND. Insert Transpose(NC(D)HW->N(D)HWC) before Reshape
        for reinterp_shape_node_id in graph.get_nodes_with_attributes(reinterp_shape=True):
            reinterp_shape_node = Node(graph, reinterp_shape_node_id)
            assert 0 in reinterp_shape_node.in_nodes(), 'Node {} does not have 0 input. \n{}'.format(
                reinterp_shape_node_id, graph.dump_graph_for_graphviz())
            input_shape = reinterp_shape_node.in_node(0).shape
            if not is_input_data_in_correct_layout(reinterp_shape_node, 0) and len(input_shape) >= 4:
                order_const = Const(graph, {'value': PermuteAttrs().get_nchw_to_nhwc_permutation(len(input_shape)).perm
                                            }).create_node()
                permute_node = Transpose(graph,
                                         {'name': reinterp_shape_node.in_port(0).get_source().node.name + '/Transpose'
                                          }).create_node()
                reinterp_shape_node.in_port(0).get_connection().insert_node(permute_node)
                order_const.out_port(0).connect(permute_node.in_port(1))
                order_const.infer(order_const)

                # do not infer the Transpose node because it should have input data node in NCHW layout (but currently
                # it is NHWC because data node attributes has not been permuted yet) and produce output in NHWC layout
                # (which is true at this moment)
                permute_node['need_shape_inference'] = False
                # mark the Transpose output data node having correct layout so it's shape will not be permuted
                mark_output_as_in_correct_layout(permute_node, 0)

                # keep the reinterp_shape_node in NHWC layout
                mark_input_as_in_correct_layout(reinterp_shape_node, 0)
                mark_input_as_in_correct_layout(reinterp_shape_node, 1)

        # reshape from ND -> 4D-5D. Insert Transpose(N(D)HWC->NC(D)HW) after Reshape
        for reinterp_shape_node_id in graph.get_nodes_with_attributes(reinterp_shape=True):
            reinterp_shape_node = Node(graph, reinterp_shape_node_id)
            assert 0 in reinterp_shape_node.out_nodes(), 'Node {} does not have 0 output. \n{}'.format(
                reinterp_shape_node_id, graph.dump_graph_for_graphviz())
            output_shape = reinterp_shape_node.out_node(0).shape
            if not is_output_data_in_correct_layout(reinterp_shape_node, 0) and len(output_shape) >= 4:
                order_const = Const(graph, {
                    'value': PermuteAttrs().get_nhwc_to_nchw_permutation(len(output_shape)).perm}).create_node()
                permute_node = Transpose(graph, {'name': reinterp_shape_node.id + '/Transpose'}).create_node()
                reinterp_shape_node.out_port(0).get_connection().insert_node(permute_node)
                order_const.out_port(0).connect(permute_node.in_port(1))

                # the Reshape and Transpose operations should work in original (NHWC layout) so the Transpose
                # will convert it to the NCHW
                mark_input_as_in_correct_layout(permute_node, 0)
                mark_input_as_in_correct_layout(permute_node, 1)
                # do not set Transpose output data node 'correct_data_layout' attribute so the data node shape will be
                # permuted

                # keep the reinterp_shape_node in NHWC layout
                mark_output_as_in_correct_layout(reinterp_shape_node, 0)
                mark_input_as_in_correct_layout(reinterp_shape_node, 1)

                # do not re-infer the Transpose node because it output data node should be in NHWC layout to make the
                # rest of the graph consistent
                permute_node['need_shape_inference'] = False

        # TODO remove the following line when the unified pipeline will be for back transformations
        graph_clean_up_tf(graph)


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
