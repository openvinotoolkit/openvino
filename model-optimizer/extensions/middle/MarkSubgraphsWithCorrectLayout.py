"""
 Copyright (C) 2018-2020 Intel Corporation

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
from collections import deque

from typing import Set

from extensions.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose, \
    mark_as_correct_data_layout
from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
from extensions.middle.pass_separator import PostMiddleStart
from mo.graph.graph import Graph, Node
from mo.graph.port import Port
from mo.middle.replacement import MiddleReplacementPattern


class MarkSubGraphsWithCorrectLayout(MiddleReplacementPattern):
    """
    The transformation looks for the layout agnostic operations which does not have a layout (NCHW or NHWC) and makes
    necessary changes to infer the part of the topology in the original layout:
    1. Prevents from adding Transpose operations before and after "reinterp_shape" like operations which change rank of
    the input and output tensors of this layout agnostic op.
    2. Disable attributes permutation for all intermediate ops between these "reinterp_shape" nodes.

    For now the transformation is triggered for MatMul operation only getting input as 4D or 5D tensors.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NHWC']
    op_conditions = [lambda n: n.soft_get('op') == 'MatMul' and
                               any([len(port.data.get_shape()) in (4, 5) for port in n.in_ports().values()]),
                     ]

    def run_after(self):
        return [PostMiddleStart]

    def run_before(self):
        return [InsertLayoutPropagationTranspose]

    @staticmethod
    def get_input_nodes(node: Node):
        return [src_port.get_source().node for src_port in node.in_ports().values()]

    @staticmethod
    def get_output_nodes(node: Node):
        result = []
        for out_port in node.out_ports().values():
            for dest_port in out_port.get_destinations():
                result.append(dest_port.node)
        return result

    def bfs(self, start_nodes: list, visited: set, condition: callable = None, forward: bool = True):
        """
        The function performs BFS starting from selected nodes in forward or backward direction adding nodes by an
        optional condition
        :param start_nodes: Nodes to start search from
        :param visited: set of already visited nodes where traversing should not happen
        :param condition: function getting a Node as input and returning whether the node should be included into the
        resukt or not. If the value is None then the node is added unconditionally.
        :param forward: boolean flag specifying the traverse direction
        :return: the list of Nodes visited
        """
        assert visited is not None, 'The "visited" set must be defined'
        assert start_nodes is not None and len(start_nodes) != 0, 'The list of start nodes must be specified'

        result = list()
        d = deque(start_nodes)
        while len(d) != 0:
            cur_node = d.popleft()
            result.append(cur_node)
            visited.add(cur_node)
            if forward:
                next_nodes = self.get_output_nodes(cur_node)
            else:
                next_nodes = self.get_input_nodes(cur_node)
            for next_node in next_nodes:
                if next_node not in visited and (condition is None or condition(next_node)):
                    d.append(next_node)
        return result

    def find_and_replace_pattern(self, graph: Graph):
        visited = set()
        marked_nodes = set()
        condition_forward = lambda n: not InsertLayoutPropagationTranspose.is_nhwc_to_nchw_transpose_needed(n)
        condition_backward = lambda n: not InsertLayoutPropagationTranspose.is_nchw_to_nhwc_transpose_needed(n)
        for node_condition in self.op_conditions:
            for node in graph.get_op_nodes():
                if node_condition(node):
                    log.debug('Detected node "{}" as a node which should be executed in the original layout'
                              ''.format(node.soft_get('name', node.id)))
                    forward_visited_nodes = self.bfs([node], visited, condition_forward, True)
                    backward_visited_nodes = self.bfs([node], visited, condition_backward, False)

                    # find "reinterp_shape" like ops which change rank of input to 4D or 5D from smaller dimensions
                    for back_node in backward_visited_nodes:
                        for input_node in self.get_input_nodes(back_node):
                            if input_node not in backward_visited_nodes and not condition_forward(input_node):
                                marked_nodes.add(input_node)

                    # find "reinterp_shape" like ops which change rank of input from 4D or 5D to smaller dimensions
                    for forward_node in forward_visited_nodes:
                        for output_node in self.get_output_nodes(forward_node):
                            if output_node not in forward_visited_nodes and not condition_backward(output_node):
                                marked_nodes.add(output_node)

                    marked_nodes.update(forward_visited_nodes + backward_visited_nodes)

        if len(marked_nodes):
            log.debug('The following nodes will be executed in the original layout: {}'
                      ''.format([n.soft_get('name', n.id) for n in marked_nodes]))

            # mark all matched nodes as in correct layout and disable attributes permutation for them
            for visited_node in marked_nodes:
                mark_as_correct_data_layout(visited_node)
                visited_node['nchw_layout'] = True

        for node in self.get_ports_and_nodes_on_weights(graph)[1]:
            mark_as_correct_data_layout(node)
            node['nchw_layout'] = True
            if node.soft_get('type') == 'Const':  # WA for Const op deletion during clean_up
                node.out_node()['nchw_layout'] = True

        for node in self.get_ports_and_nodes_on_shape_subgraphs(graph)[1]:
            mark_as_correct_data_layout(node)
            node['nchw_layout'] = True
            if node.soft_get('type') == 'Const':  # WA for Const op deletion during clean_up
                node.out_node()['nchw_layout'] = True

    @staticmethod
    def walk_up_from_in_ports_to_out_ports(in_ports: Set[Port], out_ports: Set[Port],
                                           visited_ports: Set[Port] = None, visited_nodes: Set[Node] = None):
        """"
        Returns all intermediate ports and nodes of such a sub-graph:

            out_ports
            |        |
           \/       \/
            .   .   .
            |        |
           \/       \/
            in_ports
        """
        if visited_ports is None:
            visited_ports = set()
        if visited_nodes is None:
            visited_nodes = set()

        deque_of_in_ports = deque(in_ports)
        while len(deque_of_in_ports):
            in_port = deque_of_in_ports.popleft()
            source_node = in_port.get_source().node
            if in_port in visited_ports:  # do not check visited_nodes as search is based on ports
                continue
            visited_ports.update({in_port, in_port.get_source()})
            if in_port.get_source() in out_ports:  # reached source marked to stop the search
                if not len(in_port.get_source().node.in_ports()):  # for Constants and Parameters to be visited
                    visited_nodes.add(in_port.get_source().node)
                continue
            deque_of_in_ports.extend([port for port in source_node.in_ports().values() if not port.disconnected()])
            visited_nodes.add(source_node)
        return visited_ports, visited_nodes

    @staticmethod
    def get_ports_and_nodes_on_weights(graph):
        get_weights_port_index = lambda node: node.weights_index if node.has_valid('weights_index') else 1
        weighted_layer_type_to_in_weights_port = {
            'Convolution': get_weights_port_index,
            'DeformableConvolution': get_weights_port_index,
            'Deconvolution': get_weights_port_index,
            'BinaryConvolution': get_weights_port_index,
        }
        nodes = graph.get_op_nodes()
        weighted_types = list(weighted_layer_type_to_in_weights_port.keys())

        # collect all input ports with weights
        weight_ports = set()
        start_ports = set()
        for node in nodes:
            node_type = node.soft_get('type', 'unknown')
            if node_type not in weighted_types:
                if node_type in ['Const', 'Parameter', 'ShapeOf']:
                    start_ports.add(node.out_port(0))
                continue
            weight_port_idx = weighted_layer_type_to_in_weights_port[node_type](node)
            assert node.is_in_port_connected(weight_port_idx), \
                'Unexpected port configuration of {} node with name=`{}`'.format(node_type,
                                                                                 node.soft_get('name', node.id))
            weight_ports.add(node.in_port(weight_port_idx))

        # collect all sub-graphs that start with Constant/Parameter/ShapeOf and end at in_port as weights
        ports, nodes = MarkSubGraphsWithCorrectLayout.walk_up_from_in_ports_to_out_ports(weight_ports, start_ports)
        return ports, nodes

    @staticmethod
    def get_ports_and_nodes_on_shape_subgraphs(graph):
        shape_sources = {shape_of.out_port(0) for shape_of in graph.get_op_nodes(type='ShapeOf')}
        end_points = LayoutChangeForConstantShapePaths().find_shape_subgraph_endpoints(
            [shape.out_port(0) for shape in graph.get_op_nodes(type='ShapeOf')])
        ports, nodes = MarkSubGraphsWithCorrectLayout.walk_up_from_in_ports_to_out_ports(end_points, shape_sources)
        return ports, nodes
