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

import numpy as np

from extensions.middle.ApplyNHWCtoNCHWpermutation import ApplyNHWCtoNCHWpermutation
from extensions.middle.InsertLayoutPropagationTransposes import is_input_data_in_correct_layout, \
    is_output_data_in_correct_layout
from extensions.middle.LayoutChangeForConstantShapePaths import LayoutChangeForConstantShapePaths
from extensions.middle.pass_separator import PostMiddleStart
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern
from mo.utils.error import Error


class ApplyPermutation(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True
    # can't be turned on for Kaldi until permutation logic will be aligned
    graph_condition = [lambda graph: graph.graph['fw'] != 'kaldi']

    def run_after(self):
        return [ApplyNHWCtoNCHWpermutation, PostMiddleStart]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        self.merge_nodes_permutations(graph)
        self.permute_data_nodes_attrs(graph)
        self.permute_op_nodes_attrs(graph)
        self.shape_of_sub_graph_reinference(graph)
        self.permute_input_data(graph)

    @staticmethod
    def merge_nodes_permutations(graph: Graph):
        # Iterate over all data nodes and check all permutations for similarity
        # In case of equal permutations, this permutation will be set as attribute for data node
        # otherwise exception will be raised
        for node in graph.nodes():
            node = Node(graph, node)
            if node.kind != 'data':
                continue

            permutations = []

            # Get all permutations from in edges
            for in_node in node.in_nodes():
                edge_attrs = node.graph.get_edge_data(in_node.id, node.id)[0]
                if 'permutation' in edge_attrs:
                    permutations.append(edge_attrs['permutation'])

            # Get all permutations from out edges
            for out_node in node.out_nodes():
                edge_attrs = node.graph.get_edge_data(node.id, out_node.id)[0]
                if 'permutation' in edge_attrs:
                    permutations.append(edge_attrs['permutation'])

            # Check that all permutations are equal
            final_permutations = []
            for p in permutations:
                if p is not None:
                    final_permutations.append(p.perm)
                else:
                    final_permutations.append(int64_array(np.arange(node.shape.size)))

            if len(final_permutations) == 0:
                continue

            if not all([np.array_equal(final_permutations[0], perm) for perm in final_permutations]):
                raise Error('Permutations requested for {} data node are not equal! List of permutations: {}'
                            ''.format(node.name, [p.perm for p in permutations]))

            assert not node.has_valid('permutation') or np.array_equal(node.permutation, permutations[0])
            node['permutation'] = permutations[0]

    @staticmethod
    def permute_data_nodes_attrs(graph: Graph):
        # Iterate over all data nodes and apply permutation if exists
        for node in graph.get_data_nodes():
            if not node.has_valid('permutation'):
                continue

            if len(
                    node.in_nodes()) != 0:  # there are data nodes without input operation node inside the tensor iterator
                edge_attrs = graph.get_edge_data(node.in_node(0).id, node.id)[0]
                if is_output_data_in_correct_layout(node.in_node(0), edge_attrs['out']):
                    log.debug('Do not permute data node attrs for node "{}" output port "{}"'.format(node.in_node(0).id,
                                                                                                     edge_attrs['out']))
                    continue

            # Apply permutation for shape and value if exists
            if len(node.permutation.perm) == 0:
                continue
            node.shape = np.array(node.shape)[node.permutation.perm]
            if node.has_valid('value'):
                assert len(node.value.shape) == len(node.permutation.perm), \
                    'Node {} has shape {} and permutation {} that does not match. Their lengths should be equal' \
                    ''.format(node.name, node.value.shape, node.permutation.perm)
                node.value = np.array(node.value.transpose(node.permutation.perm))

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
        if graph.graph['layout'] != 'NHWC':
            return
        for node in graph.get_op_nodes():
            input_permutations = [(in_port, edge_attrs['input_permutation']) for in_port, edge_attrs in
                                  node.in_edges().items() if edge_attrs.get('input_permutation') is not None]
            for in_port, input_perm in input_permutations:
                permutation, port_info = input_perm
                direction, port = port_info.split(':')
                port = int(port)
                port_to_check = node.in_port(port) if direction == 'input' else node.out_port(port)
                if not is_input_data_in_correct_layout(node, in_port) and len(port_to_check.data.get_shape()) >= 4:
                    permutation(node, port_info, in_port)
        graph.graph['layout'] = 'NCHW'

    @staticmethod
    def shape_of_sub_graph_reinference(graph: Graph):
        """
        After layout permutation (shape change in data nodes) shape sub-graphs contain values in the old layout
        To change that we execute full partial inference on the shape-of sub-graphs
        """
        shape_ops = graph.get_op_nodes(op='ShapeOf')
        for shape in shape_ops:
            shape.infer(shape)
        LayoutChangeForConstantShapePaths().find_shape_subgraph_endpoints(
            [shape.out_port(0) for shape in shape_ops], None, lambda in_port: in_port.node.infer(in_port.node))
