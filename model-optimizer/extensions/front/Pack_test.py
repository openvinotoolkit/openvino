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

import unittest

import numpy as np
from generator import generator, generate

from extensions.front.Pack import Pack
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

nodes_attributes = {
    'placeholder_0': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_2': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_3': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    # Pack operation
    'pack': {'axis': None, 'type': None, 'kind': 'op', 'op': 'Pack'},
    # Test operation
    'last': {'type': None, 'value': None, 'kind': 'op', 'op': None},
    # Unsqueeze, Concat and Const operations
    'const_1': {'value': None, 'type': None, 'kind': 'op', 'op': 'Const'},
    'Unsqueeze_0': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'Unsqueeze_1': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'Unsqueeze_2': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'Unsqueeze_3': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
    'Unsqueeze_0_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': None, 'value': None},
    'Unsqueeze_1_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': None, 'value': None},
    'Unsqueeze_2_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': None, 'value': None},
    'Unsqueeze_3_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'shape': None, 'value': None},
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
                graph_ref_edges.append(('placeholder_{}'.format(j), 'Unsqueeze_{}'.format(i + j)))
                graph_ref_edges.append(('Unsqueeze_{}'.format(i + j), 'concat_1'))
        graph_ref_edges.append(('concat_1', 'last'))

        update_graph_ref_attributes = {}
        for i in range(num_placeholders):
            update_graph_ref_attributes['placeholder_{}'.format(i)] = {'shape': np.array([1, 227, 227, 3])}
        for i in range(num_inputs):
            graph_ref_edges.append(('Unsqueeze_{}_axis'.format(i), 'Unsqueeze_{}'.format(i)))
            update_graph_ref_attributes['Unsqueeze_{}_axis'.format(i)] = {'shape': int64_array([1]),
                                                                          'value': int64_array([axis])}
        update_graph_ref_attributes['concat_1'] = {'axis': axis}

        graph_ref = build_graph(nodes_attributes, graph_ref_edges, update_graph_ref_attributes,
                                nodes_with_edges_only=True)

        graph.stage = 'front'

        replacer = Pack()
        replacer.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
