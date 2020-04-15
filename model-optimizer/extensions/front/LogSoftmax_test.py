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

from extensions.front.LogSoftmax import LogSoftmaxFrontReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op, result, connect

nodes = {
    **regular_op('input', {'type': 'Parameter'}),
    **regular_op('logsoftmax', {'type': None, 'op': 'LogSoftmax', 'axis': -2, 'name': 'my_logsoftmax'}),
    **result('output'),
}
edges = [
    ('input', 'logsoftmax'),
    ('logsoftmax', 'output'),
]


class LogSoftmaxReplacerTest(unittest.TestCase):
    def test_1(self):
        graph = build_graph(nodes, edges)

        graph_ref = build_graph({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('log', {'op': 'Log', 'type': 'Log'}),
            **regular_op('softmax', {'op': 'SoftMax', 'type': 'SoftMax', 'axis': -2}),
            **result('output'),
        },
            [
                ('input', 'softmax'),
                ('softmax', 'log'),
                ('log', 'output'),
            ])

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        LogSoftmaxFrontReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.get_op_nodes(op='Log')[0].name == 'my_logsoftmax')

    def test_2(self):
        graph = build_graph(nodes, edges)

        graph_ref = build_graph({
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('log', {'op': 'Log', 'type': 'Log'}),
            **regular_op('softmax', {'op': 'SoftMax', 'type': 'SoftMax', 'axis': -2}),
            **result('output'),
        },
            [
                ('input', 'softmax'),
                ('softmax', 'log'),
                ('log', 'output'),
            ])

        graph.graph['layout'] = 'NHWC'
        graph.stage = 'front'

        LogSoftmaxFrontReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.get_op_nodes(op='Log')[0].name == 'my_logsoftmax')
