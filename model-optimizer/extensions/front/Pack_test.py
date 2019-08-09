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

import unittest

import numpy as np

from extensions.front.Pack import Pack
from mo.utils.unittest.graph import build_graph, compare_graphs

from generator import generator, generate

nodes_attributes = {
    'placeholder_0': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # Pack operation
    'pack': {'axis': None, 'type': None, 'kind': 'op', 'op': 'Pack'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # ExpandDims, Concat and Const operations
    'const_1': {'value': None, 'type': None, 'kind': 'op', 'op': 'Const'},
    'ExpandDims_0': {'expand_axis': None, 'type': None, 'kind': 'op', 'op': 'ExpandDims'},
    'ExpandDims_1': {'expand_axis': None, 'type': None, 'kind': 'op', 'op': 'ExpandDims'},
    'ExpandDims_2': {'expand_axis': None, 'type': None, 'kind': 'op', 'op': 'ExpandDims'},
    'ExpandDims_3': {'expand_axis': None, 'type': None, 'kind': 'op', 'op': 'ExpandDims'},
    'concat_1': {'axis': None, 'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
}


@generator
class PackTest(unittest.TestCase):

    @generate(*[(2, 2, 0), (3, 3, 0), (4, 4, 0), (4, 4, 1), (4, 1, 0), (4, 1, 1)])
    def test_pack_test_all(self, num_inputs: int, num_placeholders: int, axis: list):

        graph_edges = []
        for i in range(num_inputs - num_placeholders + 1):
            for j in range(num_placeholders):
                graph_edges.append(('placeholder_{}'.format(j), 'pack'))
        graph_edges.append(('pack', 'last'))

        update_graph_attributes = {}
        for i in range(num_placeholders):
            update_graph_attributes['placeholder_{}'.format(i)] = {'shape': np.array([1, 227, 227, 3])}
        update_graph_attributes['pack'] = {'axis': axis}

        graph = build_graph(nodes_attributes, graph_edges, update_graph_attributes,
                            nodes_with_edges_only=True)

        graph_ref_edges = []
        for i in range(num_inputs - num_placeholders + 1):
            for j in range(num_placeholders):
                graph_ref_edges.append(('placeholder_{}'.format(j), 'ExpandDims_{}'.format(i + j)))
                graph_ref_edges.append(('ExpandDims_{}'.format(i + j), 'concat_1'))
        graph_ref_edges.append(('concat_1', 'last'))

        update_graph_ref_attributes = {}
        for i in range(num_placeholders):
            update_graph_ref_attributes['placeholder_{}'.format(i)] = {'shape': np.array([1, 227, 227, 3])}
        for i in range(num_inputs):
            update_graph_ref_attributes['ExpandDims_{}'.format(i)] = {'expand_axis': np.array([axis])}
        update_graph_ref_attributes['concat_1'] = {'axis': axis}

        graph_ref = build_graph(nodes_attributes, graph_ref_edges, update_graph_ref_attributes,
                                nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = Pack()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
