"""
 Copyright (c) 2018-2019 Intel Corporation

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
import unittest

import numpy as np

from extensions.back.PermuteForReshape import PermuteForReshape
from mo.graph.graph import Node
from mo.ops.op import PermuteAttrs
from mo.utils.unittest.graph import build_graph_with_attrs, compare_graphs


class ReshapeToPermuteTest(unittest.TestCase):
    nodes = [
        ('input_data', {'kind': 'data', 'shape': None}),
        ('reshape', {'kind': 'op', 'op': 'Squeeze', 'type': 'Reshape', 'dim': None}),
        ('reshape_data', {'kind': 'data'}),
    ]
    edges = [
        ('input_data', 'reshape'),
        ('reshape', 'reshape_data'),
    ]

    permute_nodes = [
        ('permute', {'kind': 'op', 'op': 'Permute'}),
        ('permute_data', {'kind': 'data', 'shape': None})
    ]
    permute_edges = [
        ('input_data', 'permute'),
        ('permute', 'permute_data'),
        ('permute_data', 'reshape'),
    ]

    def test_from3D_to3D(self):
        input_shape = np.array([2, 3, 4])
        new_shape = np.array([2, 3, 4])
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('input_data', {'shape': input_shape}),
                                     ('reshape', {'dim': new_shape}),
                                     ('reshape_data', {'shape': new_shape})]
        )
        graph.graph['layout'] = 'NHWC'
        # add permute attrs to reshape
        reshape = Node(graph, 'reshape')
        PermuteAttrs.create_permute_attrs(reshape, attrs=[('dim', 'output:0')])

        tested_pattern = PermuteForReshape()
        tested_pattern.find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph, last_node='reshape_data')
        self.assertTrue(flag, resp)

    def test_from4D_to3D(self):
        input_shape = np.array([1, 2, 3, 4])
        new_shape = np.array([3, 4, 2])
        nhwc_shape = np.array([1, 3, 4, 2])
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('input_data', {'shape': input_shape}),
                                     ('reshape', {'dim': new_shape}),
                                     ('reshape_data', {'shape': new_shape})]
        )
        graph.graph['layout'] = 'NHWC'
        # add permute attrs to reshape
        reshape = Node(graph, 'reshape')
        PermuteAttrs.create_permute_attrs(reshape, attrs=[('dim', 'output:0')])

        tested_pattern = PermuteForReshape()
        tested_pattern.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + self.permute_nodes,
            edges_with_attrs=self.edges[1:] + self.permute_edges,
            update_nodes_attributes=[('input_data', {'shape': input_shape}),
                                     ('reshape', {'dim': new_shape}),
                                     ('reshape_data', {'shape': new_shape}),
                                     ('permute_data', {'shape': nhwc_shape})]
        )
        # check graphs equality
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='reshape_data')
        self.assertTrue(flag, resp)

        # check righ order in new permutation node
        permute_order = graph.node['reshape/Permute_']['order']
        self.assertTrue(np.all(permute_order == np.array([0, 2, 3, 1]))) # from NCHW to NHWC

    def test_from_5D_to_3D(self):
        input_shape = np.array([1, 2, 1, 3, 4]) #  NCDHW 1 1 3 4 2
        new_shape = np.array([3, 4, 2])
        nhwc_shape = np.array([1, 1, 3, 4, 2])
        graph = build_graph_with_attrs(
            nodes_with_attrs=self.nodes,
            edges_with_attrs=self.edges,
            update_nodes_attributes=[('input_data', {'shape': input_shape}),
                                     ('reshape', {'dim': new_shape}),
                                     ('reshape_data', {'shape': new_shape})]
        )
        graph.graph['layout'] = 'NHWC'
        # add permute attrs to reshape
        reshape = Node(graph, 'reshape')
        PermuteAttrs.create_permute_attrs(reshape, attrs=[('dim', 'output:0')])

        tested_pattern = PermuteForReshape()
        tested_pattern.find_and_replace_pattern(graph)
        graph_ref = build_graph_with_attrs(
            nodes_with_attrs=self.nodes + self.permute_nodes,
            edges_with_attrs=self.edges[1:] + self.permute_edges,
            update_nodes_attributes=[('input_data', {'shape': input_shape}),
                                     ('reshape', {'dim': new_shape}),
                                     ('reshape_data', {'shape': new_shape}),
                                     ('permute_data', {'shape': nhwc_shape})]
        )
        # check graphs equality
        (flag, resp) = compare_graphs(graph, graph_ref, last_node='reshape_data')
        self.assertTrue(flag, resp)

        # check righ order in new permutation node
        permute_order = graph.node['reshape/Permute_']['order']
        self.assertTrue(np.all(permute_order == np.array([0, 2, 3, 4, 1])))  # from NCDHW to NDHWC
