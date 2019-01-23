"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.middle.Reduce import ReduceReplacer
from mo.middle.passes.eliminate_test import build_graph
from mo.middle.passes.fusing.fuse_linear_ops_test import compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    # Placeholder layers
    'placeholder_1': {'shape': None, 'type': 'Placeholder', 'kind': 'op', 'op': 'Placeholder'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_2_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_3_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},
    'placeholder_4_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Reshape layers
    'reduce_1': {'type': 'Reduce', 'kind': 'op', 'op': 'Reduce'},
    'reduce_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Reshape layers
    'reshape_1': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},

    'reshape_2': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Pooling
    'pooling': {'type': 'Pooling', 'kind': 'op', 'op': 'Pooling'},
    'pooling_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Power
    'power': {'type': 'Power', 'kind': 'op', 'op': 'Power'},
    'power_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Concat
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},
}


class ReduceReplacerTest(unittest.TestCase):
    def test1(self):
        #   Original graph
        #   data(1,64,1)-->Reduce(axis=1,keep_dims=True)-->data(1,1,1)
        #
        #   Reference graph
        #   data(1,61,1)->Reshape(1,1,64,1)->Pool(1,1,1,1)->Reshape(1,1,1)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'reduce_1'),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 64, 1])},
                             'reduce_1': {'axis': np.array([1]), 'keep_dims': True, 'reduce_type': 'Mean'},
                             'reduce_1_data': {'shape': np.array([1, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 64, 1])},
                                 'reshape_1': {'dim': np.array([1, 1, 64, 1])},
                                 'reshape_1_data': {'shape': np.array([1, 1, 64, 1])},
                                 'pooling': {'window': np.array([1, 1, 64, 1])},
                                 'pooling_data': {'shape': np.array([1, 1, 1, 1])},
                                 'reshape_2': {'dim': np.array([1, 1, 1])},
                                 'reshape_2_data': {'shape': np.array([1, 1, 1])},
                                 }, nodes_with_edges_only=True)

        pattern = ReduceReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2(self):
        #   Original graph
        #   data(1,3,64,64)-->Reduce(axis=2,keep_dims=True)-->data(1,3,1,64)
        #
        #   Reference graph
        #   data(1,3,64,64)->Reshape->Pool(1,3,1,64)->Reshape(1,3,1,64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'reduce_1'),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'reduce_1': {'axis': np.array([2]), 'keep_dims': True, 'reduce_type': 'Mean'},
                             'reduce_1_data': {'shape': np.array([1, 3, 1, 64])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'reshape_1': {'dim': np.array([1, 3, 64, 64])},
                                 'reshape_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'pooling': {'window': np.array([1, 1, 64, 1])},
                                 'pooling_data': {'shape': np.array([1, 3, 1, 64])},
                                 'reshape_2': {'dim': np.array([1, 3, 1, 64])},
                                 'reshape_2_data': {'shape': np.array([1, 3, 1, 64])},
                                 }, nodes_with_edges_only=True)

        pattern = ReduceReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test3(self):
        #   Original graph
        #   data(1,3,64,64)-->Reduce(axis=[2,3],keep_dims=True)-->data(1,3,1,1)
        #
        #   Reference graph
        #   data(1,3,64,64)->Reshape->Pool(1,3,1,1)->Reshape(1,3,1,1)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'reduce_1'),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                             'reduce_1': {'axis': np.array([2, 3]), 'keep_dims': True, 'reduce_type': 'Mean'},
                             'reduce_1_data': {'shape': np.array([1, 3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 3, 64, 64])},
                                 'reshape_1': {'dim': np.array([1, 3, 64 * 64, 1])},
                                 'reshape_1_data': {'shape': np.array([1, 3, 64 * 64, 1])},
                                 'pooling': {'window': np.array([1, 1, 64 * 64, 1])},
                                 'pooling_data': {'shape': np.array([1, 3, 1, 1])},
                                 'reshape_2': {'dim': np.array([1, 3, 1, 1])},
                                 'reshape_2_data': {'shape': np.array([1, 3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        pattern = ReduceReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test4(self):
        #   Original graph
        #   data(2,3,64,64)-->Reduce(axis=[1,2,3],keep_dims=False)-->data(2)
        #
        #   Reference graph
        #   data(2,3,64,64)->Reshape(2,1,3*64*64,1)->Pool(2,1,1,1)->Reshape(2)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'reduce_1'),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([2, 3, 64, 64])},
                             'reduce_1': {'axis': np.array([1, 2, 3]), 'keep_dims': False, 'reduce_type': 'Mean'},
                             'reduce_1_data': {'shape': np.array([2])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([2, 3, 64, 64])},
                                 'reshape_1': {'dim': np.array([2, 1, 3 * 64 * 64, 1])},
                                 'reshape_1_data': {'shape': np.array([2, 1, 3 * 64 * 64, 1])},
                                 'pooling': {'window': np.array([1, 1, 3 * 64 * 64, 1])},
                                 'pooling_data': {'shape': np.array([2, 1, 1, 1])},
                                 'reshape_2': {'dim': np.array([2])},
                                 'reshape_2_data': {'shape': np.array([2])},
                                 }, nodes_with_edges_only=True)

        pattern = ReduceReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test5(self):
        #   Original graph
        #   data(1, 16, 64, 64, 64, 4)-->Reduce(axis=[5],keep_dims=False)-->data(1, 16, 64, 64, 64)
        #
        #   Reference graph
        #   data(1, 16, 64, 64, 64, 4)->Reshape(1*16*64*64, 64, 4, 1)->Pool(1, 1, 4, 1)->Reshape(1, 16, 64, 64, 64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'reduce_1'),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 16, 64, 64, 64, 4])},
                             'reduce_1': {'axis': np.array([5]), 'keep_dims': False, 'reduce_type': 'max'},
                             'reduce_1_data': {'shape': np.array([1, 16, 64, 64, 64])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 16, 64, 64, 64, 4])},
                                 'reshape_1': {'dim': np.array([65536, 64, 4, 1])},
                                 'reshape_1_data': {'shape': np.array([65536, 64, 4, 1])},
                                 'pooling': {'window': np.array([1, 1, 4, 1])},
                                 'pooling_data': {'shape': np.array([65536, 64, 1, 1])},
                                 'reshape_2': {'dim': np.array([1, 16, 64, 64, 64])},
                                 'reshape_2_data': {'shape': np.array([1, 16, 64, 64, 64])},
                                 }, nodes_with_edges_only=True)

        pattern = ReduceReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test6(self):
        #   Original graph
        #   data(1,64,1)-->Reduce(axis=-2,keep_dims=True, reduce_type=Sum)-->data(1,1,1)
        #
        #   Reference graph
        #   data(1,61,1)->Reshape(1,1,64,1)->Pool(1,1,1,1)->Reshape(1,1,1)->Power(scale=64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1_data', 'reduce_1'),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': np.array([1, 64, 1])},
                             'reduce_1': {'axis': np.array([-2]), 'keep_dims': True, 'reduce_type': 'Sum'},
                             'reduce_1_data': {'shape': np.array([1, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'power'),
                                 ('power', 'power_data'),
                                 ('power_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': np.array([1, 64, 1])},
                                 'reshape_1': {'dim': np.array([1, 1, 64, 1])},
                                 'reshape_1_data': {'shape': np.array([1, 1, 64, 1])},
                                 'pooling': {'window': np.array([1, 1, 64, 1])},
                                 'pooling_data': {'shape': np.array([1, 1, 1, 1])},
                                 'reshape_2': {'dim': np.array([1, 1, 1])},
                                 'reshape_2_data': {'shape': np.array([1, 1, 1])},
                                 'power': {'scale': 64.0},
                                 'power_data': {'shape': np.array([1, 1, 1])},
                                 }, nodes_with_edges_only=True)

        pattern = ReduceReplacer()
        pattern.find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)
