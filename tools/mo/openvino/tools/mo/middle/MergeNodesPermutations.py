# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.middle.ApplyNHWCtoNCHWpermutation import ApplyNHWCtoNCHWpermutation
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.utils.error import Error


class MergeNodesPermutations(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        return [ApplyNHWCtoNCHWpermutation]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        self.merge_nodes_permutations(graph)

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

            final_permutations = []
            for p in permutations:
                if p is not None:
                    final_permutations.append(p.perm)
                else:
                    final_permutations.append(int64_array(np.arange(node.shape.size)))

            if len(final_permutations) == 0:
                continue

            # Check that all permutations are equal
            if not all([np.array_equal(final_permutations[0], perm) for perm in final_permutations]):
                raise Error('Permutations requested for {} data node are not equal! List of permutations: {}'
                            ''.format(node.name, [p.perm for p in permutations]))

            assert not node.has_valid('permutation') or np.array_equal(node.permutation, permutations[0])
            node['permutation'] = permutations[0]
