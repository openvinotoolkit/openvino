# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.middle.InsertLayoutPropagationTransposes import is_input_data_in_correct_layout, \
    is_output_data_in_correct_layout
from openvino.tools.mo.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
from openvino.tools.mo.middle.PreserveRuntimeInfo import PreserveRuntimeInfo
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.graph.perm_inputs import get_node_with_permutation
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.utils.error import Error


class ApplyPermutation(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    # can't be turned on for Kaldi until permutation logic will be aligned
    graph_condition = [lambda graph: graph.graph['fw'] != 'kaldi']

    def run_after(self):
        return [PreserveRuntimeInfo]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        self.permute_data_nodes_attrs(graph)
        self.permute_op_nodes_attrs(graph)
        self.shape_of_sub_graph_reinference(graph)
        self.permute_input_data(graph)
        graph.graph['layout'] = 'NCHW'

    @staticmethod
    def permute_data_nodes_attrs(graph: Graph):
        # Iterate over all data nodes and apply permutation if exists
        for node in graph.get_data_nodes():
            if not node.has_valid('permutation') or \
                    all([attrs.get('input_permutation', False) for u, v, attrs in graph.out_edges(node.id, data=True)]):
                continue

            if len(node.in_nodes()) != 0:  # there are data nodes without input operation node inside the TensorIterator
                edge_attrs = graph.get_edge_data(node.in_node(0).id, node.id)[0]
                if is_output_data_in_correct_layout(node.in_node(0), edge_attrs['out']):
                    log.debug('Do not permute data node attrs for node "{}" output port "{}"'.format(node.in_node(0).id,
                                                                                                     edge_attrs['out']))
                    continue

            # Apply permutation for shape and value if exists
            if len(node.permutation.perm) == 0:
                continue
            node.shape = shape_array(node.shape)[node.permutation.perm]
            if node.has_valid('value'):
                assert len(node.value.shape) == len(node.permutation.perm), \
                    'Node {} has shape {} and permutation {} that does not match. Their lengths should be equal' \
                    ''.format(node.name, node.value.shape, node.permutation.perm)
                node.value = mo_array(node.value.transpose(node.permutation.perm))

    @staticmethod
    def permute_op_nodes_attrs(graph: Graph):
        for node in graph.get_op_nodes():
            if node.has_valid('permute_attrs') and not node.has_and_set('nchw_layout'):
                try:
                    node.permute_attrs.permute_attrs(node)
                except Exception as e:
                    raise Error('Can\'t permute attrs for node {}. Error message: {}'.format(node.id, e))

    @staticmethod
    def permute_input_data(graph: Graph):
        for node in graph.get_op_nodes():
            input_permutations = [(in_port, edge_attrs['input_permutation']) for in_port, edge_attrs in
                                  node.in_edges().items() if edge_attrs.get('input_permutation') is not None]
            for in_port, input_perm in input_permutations:
                permutation, port_info, check_shape = input_perm
                direction, port = port_info.split(':')
                port = int(port)
                port_to_check = node.in_port(port) if direction == 'input' else node.out_port(port)
                permutation_data_node = get_node_with_permutation(node, port_info)

                if permutation_data_node.has_and_set('permutation') and \
                        not is_input_data_in_correct_layout(node, in_port) and check_shape(port_to_check):
                    permutation(node, port_info, in_port)
            if node.has_and_set('need_shape_inference'):
                node.infer(node)
                node.need_shape_inference = False

    @staticmethod
    def shape_of_sub_graph_reinference(graph: Graph):
        """
        After layout permutation (shape change in data nodes) shape sub-graphs contain values in the old layout
        To change that we execute full partial inference on the shape-of sub-graphs
        """
        shape_ops = graph.get_op_nodes(op='ShapeOf')
        for shape in shape_ops:
            shape.infer(shape)

        def reinfer_once(in_port: Port):
            node = in_port.node
            if not node.soft_get('reinferred', False):
                node.infer(node)
                node['reinferred'] = True

        LayoutChangeForConstantShapePaths().find_shape_subgraph_endpoints(
            out_ports=[shape.out_port(0) for shape in shape_ops], action=reinfer_once)
