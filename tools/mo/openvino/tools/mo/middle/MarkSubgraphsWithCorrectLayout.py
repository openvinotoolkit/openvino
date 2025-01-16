# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from collections import deque
from typing import Set

from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import InsertLayoutPropagationTranspose, \
    mark_as_correct_data_layout, mark_output_as_in_correct_layout, mark_input_as_in_correct_layout
from openvino.tools.mo.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
from openvino.tools.mo.middle.pass_separator import PostMiddleStart
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.graph.perm_inputs import PermuteInputs
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class MarkSubGraphsWithCorrectLayout(MiddleReplacementPattern):
    """
    The transformation looks for the layout agnostic operations which does not have a layout (NCHW or NHWC) and makes
    necessary changes to infer the part of the topology in the original layout:
    1. Prevents from adding Transpose operations before and after "reinterp_shape" like operations which change rank of
    the input and output tensors of this layout agnostic op.
    2. Disable attributes permutation for all intermediate ops between these "reinterp_shape" nodes.
    3. Marks nodes along the weight path of convolutions as in correct layout to not permute them from NHWC to NCHW.
    The latest is needed for TF NCHW graphs as well. In Conv/Deconv infer functions "set_permutation()"
    ads "permutation" attr to weights data node even for NCHW, it is needed to permute Conv weights from the
    original TF layout into OV even for NCHW graphs. Therefore for TF models
    to prevent unwarranted permutations need to mark weights path as having correct layout even for NCHW graphs.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']
    op_conditions = [lambda n: n.soft_get('op') == 'MatMul' and
                               any([len(port.data.get_shape()) in (4, 5) for port in n.in_ports().values()]),
                     ]

    def run_after(self):
        return [PostMiddleStart]

    def run_before(self):
        return [InsertLayoutPropagationTranspose]

    @staticmethod
    def get_input_nodes(node: Node):
        return [src_port.get_source().node for src_port in node.in_ports().values() if not src_port.disconnected()]

    @staticmethod
    def get_output_nodes(node: Node):
        result = []
        for out_port in node.out_ports().values():
            if not out_port.disconnected():
                for dest_port in out_port.get_destinations():
                    result.append(dest_port.node)
        return result

    @staticmethod
    def bfs(start_nodes: list, visited: set, condition: callable = None, forward: bool = True):
        """
        The function performs BFS starting from selected nodes in forward or backward direction adding nodes by an
        optional condition
        :param start_nodes: Nodes to start search from
        :param visited: set of already visited nodes where traversing should not happen
        :param condition: function getting a Node as input and returning whether the node should be included into the
        result or not. If the value is None then the node is added unconditionally.
        :param forward: boolean flag specifying the traverse direction
        :return: the list of Nodes visited
        """
        assert visited is not None, 'The "visited" set must be defined'
        assert start_nodes is not None, 'The list of start nodes must be specified'

        result = list()
        d = deque(start_nodes)
        while len(d) != 0:
            cur_node = d.popleft()
            result.append(cur_node)
            visited.add(cur_node)
            if forward:
                next_nodes = MarkSubGraphsWithCorrectLayout.get_output_nodes(cur_node)
            else:
                next_nodes = MarkSubGraphsWithCorrectLayout.get_input_nodes(cur_node)
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

        _, nodes_weigths, nodes_in_weights = self.get_ports_and_nodes_on_weights(graph)
        for node in nodes_weigths:
            if node in nodes_in_weights:
                for ind, port in node.in_ports().items():
                    if ind not in nodes_in_weights[node]:
                        mark_input_as_in_correct_layout(node, ind)
                for ind, port in node.out_ports().items():
                    mark_output_as_in_correct_layout(node, ind)
            else:
                mark_as_correct_data_layout(node)
            node['nchw_layout'] = True

        for node in self.get_ports_and_nodes_on_shape_subgraphs(graph)[1]:
            mark_as_correct_data_layout(node)
            node['nchw_layout'] = True

    @staticmethod
    def get_weighted_layer_type_to_in_weights_port():
        get_weights_port_index = lambda node: node.weights_index if node.has_valid('weights_index') else 1
        weighted_layer_type_to_in_weights_port = {
            'Convolution': get_weights_port_index,
            'DeformableConvolution': get_weights_port_index,
            'Deconvolution': get_weights_port_index,
            'BinaryConvolution': get_weights_port_index,
        }
        return weighted_layer_type_to_in_weights_port

    @staticmethod
    def insert_permute_inputs_before_dynamic_weights_subgraph(dynamic_subgraphs: Set[Node] = None):
        """
        The function inserts permutations on input nodes in the weights subgraph
        :param dynamic_subgraphs: Set of Nodes belonging to weight path subgraphs
        :return: the list of Nodes which are inputs to weight path subgraphs
        """
        dynamic_in_nodes = dict()
        for node in dynamic_subgraphs:
            node_type = node.soft_get('type')
            if node_type not in ['Const', 'Parameter', 'ShapeOf']:
                idx_lst = list()
                for idx in [idx for idx, port in node.in_ports().items() if
                            not port.disconnected() and port.get_source().node not in dynamic_subgraphs]:
                    PermuteInputs().set_input_permutation(node.in_node(idx), node, 'input:{}'.format(idx),
                                                          'transpose_nchw_to_nhwc')
                    idx_lst.append(idx)
                if len(idx_lst):
                    dynamic_in_nodes[node] = idx_lst
        return dynamic_in_nodes

    @staticmethod
    def walk_up_from_in_ports_to_out_ports(in_ports: Set[Port], out_ports: Set[Port], port_condition=None):
        r""""
        Returns all intermediate ports and nodes of such a sub-graph:

            out_ports
            |        |
           \/       \/
            .   .   .
            |        |
           \/       \/
            in_ports
        """
        visited_ports = set()
        visited_nodes = set()

        deque_of_in_ports = deque(in_ports)
        while len(deque_of_in_ports):
            in_port = deque_of_in_ports.popleft()
            if in_port.get_source() is None:
                continue
            source_node = in_port.get_source().node
            if in_port in visited_ports:  # do not check visited_nodes as search is based on ports
                continue
            visited_ports.update({in_port, in_port.get_source()})
            if in_port.get_source() in out_ports:  # reached source marked to stop the search
                if not len(in_port.get_source().node.in_ports()):  # for Constants and Parameters to be visited
                    visited_nodes.add(in_port.get_source().node)
                continue
            for idx, port in source_node.in_ports().items():
                if not port.disconnected() and (not port_condition or port_condition(source_node, idx)):
                    deque_of_in_ports.append(port)
            visited_nodes.add(source_node)
        return visited_ports, visited_nodes

    @staticmethod
    def is_not_weight_port(node: Node, idx: int):
        w_types_to_in_port_dict = MarkSubGraphsWithCorrectLayout.get_weighted_layer_type_to_in_weights_port()
        node_type = node.soft_get('type')
        return node_type in w_types_to_in_port_dict.keys() and idx != w_types_to_in_port_dict[node_type](node)

    @staticmethod
    def get_ports_and_nodes_on_weights(graph):
        nodes = graph.get_op_nodes()

        # collect all input ports with weights
        weight_ports = set()
        result_ports = set()
        start_ports = set()
        w_types_to_in_port_dict = MarkSubGraphsWithCorrectLayout.get_weighted_layer_type_to_in_weights_port()
        for node in nodes:
            node_type = node.soft_get('type', 'unknown')
            if node_type not in w_types_to_in_port_dict.keys():
                if node_type in ['Const', 'Parameter', 'ShapeOf', 'ExtractImagePatches']:
                    start_ports.add(node.out_port(0))
                continue
            weight_port_idx = w_types_to_in_port_dict[node_type](node)
            assert node.is_in_port_connected(weight_port_idx), \
                'Unexpected port configuration of {} node with name=`{}`'.format(node_type,
                                                                                 node.soft_get('name', node.id))
            weight_ports.add(node.in_port(weight_port_idx))
        for result in graph.get_op_nodes(type='Result'):
            result_ports.update(result.in_ports().values())

        # collect all sub-graphs that start with Constant/Parameter/ShapeOf/ExtractImagePatches and end at in_port as
        # weights
        ports_w, nodes_w = MarkSubGraphsWithCorrectLayout.walk_up_from_in_ports_to_out_ports(weight_ports, start_ports)
        # collect all sub-graphs that start with Constant/Parameter/ShapeOf/ExtractImagePatches, end at Result nodes and
        # not contains branches that end as weights
        ports_d, nodes_d = MarkSubGraphsWithCorrectLayout.walk_up_from_in_ports_to_out_ports(
            result_ports, start_ports, MarkSubGraphsWithCorrectLayout.is_not_weight_port)
        nodes_dif = nodes_w.difference(nodes_d)
        nodes_in_w = MarkSubGraphsWithCorrectLayout.insert_permute_inputs_before_dynamic_weights_subgraph(nodes_dif)
        return ports_w.difference(ports_d), nodes_dif, nodes_in_w

    @staticmethod
    def get_ports_and_nodes_on_shape_subgraphs(graph):
        shape_sources = {shape_of.out_port(0) for shape_of in graph.get_op_nodes(type='ShapeOf')}
        end_points = LayoutChangeForConstantShapePaths().find_shape_subgraph_endpoints(
            [shape.out_port(0) for shape in graph.get_op_nodes(type='ShapeOf')])
        ports, nodes = MarkSubGraphsWithCorrectLayout.walk_up_from_in_ports_to_out_ports(end_points, shape_sources)
        return ports, nodes
