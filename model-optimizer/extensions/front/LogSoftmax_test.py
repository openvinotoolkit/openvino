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

from generator import generator, generate

from extensions.front.LogSoftmax import LogSoftmaxFrontReplacer
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph

graph_node_attributes = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'logsoftmax': {'type': None, 'kind': 'op', 'op': 'LogSoftmax', 'axis': -1},
    'output': {'kind': 'op', 'type': 'Result', 'op': 'Result'},
}


graph_edges = [
    ('placeholder', 'logsoftmax'),
    ('logsoftmax', 'output'),
]


graph_ref_node_attributes = {
    'placeholder': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
    'exp': {'type': 'Exp', 'kind': 'op', 'op': 'Exp'},
    'reduce_sum':  {'type': 'ReduceSum', 'kind': 'op', 'op': 'ReduceSum', 'keep_dims': True},
    'reduce_max':  {'type': 'ReduceMax', 'kind': 'op', 'op': 'ReduceMax', 'keep_dims': True},
    'log': {'type': 'Log', 'kind': 'op', 'op': 'Log'},
    'second_sub': {'type': 'Subtract', 'kind': 'op', 'op': 'Sub'},
    'reduce_sum_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': int64_array([1])},
    'reduce_max_axis': {'type': 'Const', 'kind': 'op', 'op': 'Const', 'value': None, 'shape': int64_array([1])},
    'first_sub': {'type': 'Subtract', 'kind': 'op', 'op': 'Sub'},
    'output': {'kind': 'op', 'type': 'Result', 'op': 'Result'},
}


graph_ref_edges = [
    ('placeholder', 'reduce_max', {'in': 0, 'out': 0}),
    ('placeholder', 'first_sub', {'in': 0, 'out': 0}),
    ('reduce_max', 'first_sub', {'in': 1}),
    ('reduce_max_axis', 'reduce_max', {'in': 1}),
    ('first_sub', 'exp', {'in': 0, 'out': 0}),
    ('first_sub', 'second_sub', {'in': 0, 'out': 0}),
    ('exp', 'reduce_sum', {'in': 0}),
    ('reduce_sum_axis', 'reduce_sum', {'in': 1}),
    ('reduce_sum', 'log'),
    ('log', 'second_sub', {'in': 1}),
    ('second_sub', 'output'),
]


@generator
class LogSoftmaxReplacerTest(unittest.TestCase):
    @generate(*[(-1, 'NCHW'), (-1, 'NHWC'), (0, 'NHWC'),
                (0, 'NCHW'), (2, 'NCHW'), (2, 'NHWC'),
                (-2, 'NHWC'), (-2, 'NCHW')])
    def test_logsoftmax_replacer(self, axis, layout):
        graph = build_graph(nodes_attrs=graph_node_attributes, edges=graph_edges)
        graph_ref = build_graph(nodes_attrs=graph_ref_node_attributes,
                                edges=graph_ref_edges,
                                update_attributes={
                                    'reduce_max_axis': {'value': int64_array([axis])},
                                    'reduce_sum_axis': {'value': int64_array([axis])},
                                })
        graph.graph['layout'] = layout
        graph.stage = 'front'
        LogSoftmaxFrontReplacer().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'output')
        self.assertTrue(flag, resp)

