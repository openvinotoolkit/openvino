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

from extensions.back.ReduceToPooling import ReduceReplacer, ReduceMerge
from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.eliminate import shape_inference
from mo.middle.passes.eliminate_test import build_graph
from mo.utils.ir_engine.compare_graphs import compare_graphs

# The dictionary with nodes attributes used to build various graphs. A key is the name of the node and the value is the
# dictionary with node attributes.
nodes_attributes = {
    # Placeholder layers
    'placeholder_1': {'shape': None, 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'placeholder_1_data': {'value': None, 'shape': None, 'kind': 'data', 'data_type': None},

    # Reduce layers
    'const': {'type': 'Const', 'value': None, 'kind': 'op'},
    'const_data': {'kind': 'data', 'value': None, 'shape': None},
    'reduce_1': {'type': 'Reduce', 'kind': 'op', 'op': 'Reduce'},
    'reduce_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'const_2': {'type': 'Const', 'value': None, 'kind': 'op'},
    'const_2_data': {'kind': 'data', 'value': None, 'shape': None},
    'reduce_2': {'type': 'Reduce', 'kind': 'op', 'op': 'Reduce'},
    'reduce_2_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Reshape layers
    'reshape_1': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_1_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_1_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'reshape_1_const_data': {'kind': 'data', 'value': None, 'shape': None},

    'reshape_2': {'type': 'Reshape', 'kind': 'op', 'op': 'Reshape'},
    'reshape_2_data': {'value': None, 'shape': None, 'kind': 'data'},
    'reshape_2_const': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None},
    'reshape_2_const_data': {'kind': 'data', 'value': None, 'shape': None},

    # Pooling
    'pooling': {'type': 'Pooling', 'kind': 'op', 'op': 'Pooling'},
    'pooling_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Power
    'power': {'type': 'Power', 'kind': 'op', 'op': 'AttributedPower'},
    'power_data': {'value': None, 'shape': None, 'kind': 'data'},

    # Concat
    'concat': {'type': 'Concat', 'kind': 'op', 'op': 'Concat'},

    # Result
    'result': {'type': 'Result', 'kind': 'op', 'op': 'Result'},
}


class ReduceReplacerTest(unittest.TestCase):
    def test1(self):
        #   Original graph
        #   data(1,64,1)-->Reduce(axis=1,keep_dims=True)-->data(1,1,1)
        #
        #   Reference graph
        #   data(1,64,1)->Reshape(1,1,8,8)->Pool(1,1,1,1)->Reshape(1,1,1)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1_data': {'shape': int64_array([1, 64, 1])},
                             'reduce_1': {'keep_dims': True, 'type': 'ReduceMean'},
                             'const_data': {'value': int64_array([1])},
                             'reduce_1_data': {'shape': int64_array([1, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1_data': {'shape': int64_array([1, 64, 1])},
                                 'reshape_1_const': {'value': int64_array([0, 1, 8, 8]), 'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1, 8, 8]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 8, 8])},
                                 'pooling': {'window': int64_array([1, 1, 8, 8])},
                                 'pooling_data': {'shape': int64_array([1, 1, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([0, 1, 1]), 'shape': int64_array([3])},
                                 'reshape_2_const_data': {'value': int64_array([0, 1, 1]), 'shape': int64_array([3])},
                                 'reshape_2_data': {'shape': int64_array([1, 1, 1])},
                                 }, nodes_with_edges_only=True)

        ReduceReplacer().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2(self):
        #   Original graph
        #   data(1,3,64,64)-->Reduce(axis=2,keep_dims=True)-->data(1,3,1,64)
        #
        #   Reference graph
        #   data(1,3,64,64)->Pool(1,3,1,64)->Reshape(1,3,1,64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1': {'shape': int64_array([1, 3, 64, 64])},
                             'placeholder_1_data': {'shape': int64_array([1, 3, 64, 64])},
                             'reduce_1': {'keep_dims': True, 'type': 'ReduceMean'},
                             'const_data': {'value': int64_array([2])},
                             'reduce_1_data': {'shape': int64_array([1, 3, 1, 64])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1': {'shape': int64_array([1, 3, 64, 64])},
                                 'placeholder_1_data': {'shape': int64_array([1, 3, 64, 64])},
                                 'pooling': {'window': int64_array([1, 1, 64, 1])},
                                 'pooling_data': {'shape': int64_array([1, 3, 1, 64])},
                                 'reshape_2_const': {'value': int64_array([0, 3, 1, 64]), 'shape': int64_array([4])},
                                 'reshape_2_const_data': {'value': int64_array([0, 3, 1, 64]),
                                                          'shape': int64_array([4])},
                                 'reshape_2_data': {'shape': int64_array([1, 3, 1, 64])},
                                 }, nodes_with_edges_only=True)

        ReduceReplacer().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test3(self):
        #   Original graph
        #   data(1,3,64,64)-->Reduce(axis=[2,3],keep_dims=True)-->data(1,3,1,1)
        #
        #   Reference graph
        #   data(1,3,64,64)->Pool(1,3,1,1)->Reshape(1,3,1,1)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1': {'shape': int64_array([1, 3, 64, 64])},
                             'placeholder_1_data': {'shape': int64_array([1, 3, 64, 64])},
                             'reduce_1': {'keep_dims': True, 'type': 'ReduceMean'},
                             'const_data': {'value': int64_array([2, 3])},
                             'reduce_1_data': {'shape': int64_array([1, 3, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1': {'shape': int64_array([1, 3, 64, 64])},
                                 'placeholder_1_data': {'shape': int64_array([1, 3, 64, 64])},
                                 'pooling': {'window': int64_array([1, 1, 64, 64])},
                                 'pooling_data': {'shape': int64_array([1, 3, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([0, 3, 1, 1]), 'shape': int64_array([4])},
                                 'reshape_2_const_data': {'value': int64_array([0, 3, 1, 1]),
                                                          'shape': int64_array([4])},
                                 'reshape_2_data': {'shape': int64_array([1, 3, 1, 1])},
                                 }, nodes_with_edges_only=True)

        ReduceReplacer().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test4(self):
        #   Original graph
        #   data(2,3,64,64)-->Reduce(axis=[1,2,3],keep_dims=False)-->data(2)
        #
        #   Reference graph
        #   data(2,3,64,64)->Reshape(2,1,96,128)->Pool(2,1,1,1)->Reshape(2)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1': {'shape': int64_array([2, 3, 64, 64])},
                             'placeholder_1_data': {'shape': int64_array([2, 3, 64, 64])},
                             'reduce_1': {'keep_dims': False, 'type': 'ReduceMean'},
                             'const_data': {'value': int64_array([1, 2, 3])},
                             'reduce_1_data': {'shape': int64_array([2])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1': {'shape': int64_array([2, 3, 64, 64])},
                                 'placeholder_1_data': {'shape': int64_array([2, 3, 64, 64])},
                                 'reshape_1_const': {'value': int64_array([0, 1, 96, 128]),
                                                     'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1, 96, 128]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([2, 1, 96, 128])},
                                 'pooling': {'window': int64_array([1, 1, 96, 128])},
                                 'pooling_data': {'shape': int64_array([2, 1, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_2_const_data': {'value': int64_array([0]), 'shape': int64_array([1])},
                                 'reshape_2_data': {'shape': int64_array([2])},
                                 }, nodes_with_edges_only=True)

        ReduceReplacer().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test5(self):
        #   Original graph
        #   data(1, 16, 64, 64, 64, 4)-->Reduce(axis=[5],keep_dims=False)-->data(1, 16, 64, 64, 64)
        #
        #   Reference graph
        #   data(1, 16, 64, 64, 64, 4)->Reshape(1*16*64*64, 64, 2, 2)->Pool(1, 1, 2, 2)->Reshape(1, 16, 64, 64, 64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1': {'shape': int64_array([1, 16, 64, 64, 64, 4])},
                             'placeholder_1_data': {'shape': int64_array([1, 16, 64, 64, 64, 4])},
                             'reduce_1': {'keep_dims': False, 'type': 'ReduceMax'},
                             'const_data': {'value': int64_array([5])},
                             'reduce_1_data': {'shape': int64_array([1, 16, 64, 64, 64])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'concat'),
                                 ],
                                {'placeholder_1': {'shape': int64_array([1, 16, 64, 64, 64, 4])},
                                 'placeholder_1_data': {'shape': int64_array([1, 16, 64, 64, 64, 4])},
                                 'reshape_1_const': {'value': int64_array([0, 4194304, 2, 2]),
                                                     'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([0, 4194304, 2, 2]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([1, 4194304, 2, 2])},
                                 'pooling': {'window': int64_array([1, 1, 2, 2])},
                                 'pooling_data': {'shape': int64_array([1, 4194304, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([0, 16, 64, 64, 64]),
                                                     'shape': int64_array([5])},
                                 'reshape_2_const_data': {'value': int64_array([0, 16, 64, 64, 64]),
                                                          'shape': int64_array([5])},
                                 'reshape_2_data': {'shape': int64_array([1, 16, 64, 64, 64])},
                                 }, nodes_with_edges_only=True)

        ReduceReplacer().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test6(self):
        #   Original graph
        #   data(1,64,1)-->Reduce(axis=-2,keep_dims=True, reduce_type=Sum)-->data(1,1,1)
        #
        #   Reference graph
        #   data(1,61,1)->Reshape(1,1,8,8)->Pool(1,1,1,1)->Reshape(1,1,1)->Power(scale=64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'concat'),
                             ],
                            {'placeholder_1': {'shape': int64_array([1, 64, 1])},
                             'placeholder_1_data': {'shape': int64_array([1, 64, 1])},
                             'reduce_1': {'keep_dims': True, 'type': 'ReduceSum'},
                             'const_data': {'value': int64_array([-2])},
                             'reduce_1_data': {'shape': int64_array([1, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'power'),
                                 ('power', 'power_data'),
                                 ('power_data', 'concat'),
                                 ],
                                {'placeholder_1': {'shape': int64_array([1, 64, 1])},
                                 'placeholder_1_data': {'shape': int64_array([1, 64, 1])},
                                 'reshape_1_const': {'value': int64_array([0, 1, 8, 8]), 'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1, 8, 8]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 8, 8])},
                                 'pooling': {'window': int64_array([1, 1, 8, 8])},
                                 'pooling_data': {'shape': int64_array([1, 1, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([0, 1, 1]), 'shape': int64_array([3])},
                                 'reshape_2_const_data': {'value': int64_array([0, 1, 1]), 'shape': int64_array([3])},
                                 'reshape_2_data': {'shape': int64_array([1, 1, 1])},
                                 'power': {'scale': 64.0},
                                 'power_data': {'shape': int64_array([1, 1, 1])},
                                 }, nodes_with_edges_only=True)

        ReduceReplacer().find_and_replace_pattern(graph)
        shape_inference(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test7(self):
        #   Original graph
        #   data(1,1,64,64)-->Reduce(axis=-1,keep_dims=True, reduce_type=Mean)-->Reduce(same, axis=-2)-->data(1,1,1,1)
        #
        #   Reference graph
        #   data(1,61,1)->Reshape(1,1,8,8)->Pool(1,1,1,1)->Reshape(1,1,1)->Power(scale=64)
        #
        graph = build_graph(nodes_attributes,
                            [('placeholder_1', 'placeholder_1_data'),
                             ('placeholder_1_data', 'reduce_1'),
                             ('const', 'const_data'),
                             ('const_data', 'reduce_1', {'in': 1}),
                             ('reduce_1', 'reduce_1_data'),
                             ('reduce_1_data', 'reduce_2'),
                             ('const_2', 'const_2_data'),
                             ('const_2_data', 'reduce_2', {'in': 1}),
                             ('reduce_2', 'reduce_2_data'),
                             ('reduce_2_data', 'result'),
                             ],
                            {'placeholder_1': {'shape': int64_array([1, 1, 64, 64])},
                             'placeholder_1_data': {'shape': int64_array([1, 1, 64, 64])},
                             'reduce_1': {'keep_dims': True, 'type': 'ReduceMean'},
                             'const_data': {'value': int64_array([-1])},
                             'reduce_1_data': {'shape': int64_array([1, 1, 64, 1])},
                             'reduce_2': {'keep_dims': True, 'type': 'ReduceMean'},
                             'const_2_data': {'value': int64_array([-2])},
                             'reduce_2_data': {'shape': int64_array([1, 1, 1, 1])},
                             }, nodes_with_edges_only=True)

        graph.graph['layout'] = 'NCHW'

        graph_ref = build_graph(nodes_attributes,
                                [('placeholder_1', 'placeholder_1_data'),
                                 ('placeholder_1_data', 'reshape_1'),
                                 ('reshape_1_const', 'reshape_1_const_data'),
                                 ('reshape_1_const_data', 'reshape_1'),
                                 ('reshape_1', 'reshape_1_data'),
                                 ('reshape_1_data', 'pooling'),
                                 ('pooling', 'pooling_data'),
                                 ('pooling_data', 'reshape_2'),
                                 ('reshape_2_const', 'reshape_2_const_data'),
                                 ('reshape_2_const_data', 'reshape_2'),
                                 ('reshape_2', 'reshape_2_data'),
                                 ('reshape_2_data', 'result'),
                                 ],
                                {'placeholder_1': {'shape': int64_array([1, 1, 64, 64])},
                                 'reshape_1_const': {'value': int64_array([0, 1, 8, 8]), 'shape': int64_array([4])},
                                 'reshape_1_const_data': {'value': int64_array([0, 1, 8, 8]),
                                                          'shape': int64_array([4])},
                                 'reshape_1_data': {'shape': int64_array([1, 1, 8, 8])},
                                 'pooling': {'window': int64_array([1, 1, 8, 8])},
                                 'pooling_data': {'shape': int64_array([1, 1, 1, 1])},
                                 'reshape_2_const': {'value': int64_array([0, 1, 1, 1]), 'shape': int64_array([4])},
                                 'reshape_2_const_data': {'value': int64_array([0, 1, 1, 1]),
                                                          'shape': int64_array([4])},
                                 'reshape_2_data': {'shape': int64_array([1, 1, 1, 1])},
                                 }, nodes_with_edges_only=True)

        ReduceMerge().find_and_replace_pattern(graph)
        ReduceReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)


@generator
class DimNormalizer(unittest.TestCase):
    @generate(*[
        (1, [1, 1]),
        (2, [1, 2]),
        (3, [1, 3]),
        (4, [2, 2]),
        (5, [1, 5]),
        (9, [3, 3]),
        (19, [1, 19]),
        (1000, [25, 40]),
        (1000003, [1, 1000003]),
        (1005973, [997, 1009]),
    ])
    def test_initial_reshape_dim_normalizer(self, number, expected_output):
        window = ReduceReplacer.initial_reshape_dim_normalizer(number)
        self.assertIsNotNone(window, "window is None for i={}".format(number))
        self.assertEqual(number, np.prod(window), "{} != prod({})".format(number, window))
        self.assertEqual(expected_output, window, "{} != {}".format(expected_output, window))
