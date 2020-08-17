"""
 Copyright (C) 2020 Intel Corporation

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

from extensions.front.FakeOutputResolver import FakeOutputResolver
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, result, regular_op, const


class FakeOutputResolverTest(unittest.TestCase):
    def test_one(self):
        nodes = {
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('some_op', {'type': 'SomeOp', 'name': 'some_op_name'}),
            **regular_op('fake_output', {'type': None, 'kind': 'op', 'op': 'FakeOutput', 'name': 'my_output_name'}),
            **result('result'),
        }
        edges = [('input', 'some_op'),
                 ('some_op', 'fake_output'),
                 ('fake_output', 'result'),
                 ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('input', 'some_op'),
                     ('some_op', 'result'),
                     ]

        graph_ref = build_graph(nodes, edges_ref, {'some_op': {'name': 'my_output_name'}})

        FakeOutputResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result')
        self.assertTrue(flag, resp)

    def test_multi(self):
        nodes = {
            **regular_op('input', {'type': 'Parameter'}),
            **regular_op('some_op', {'type': 'SomeOp', 'name': 'some_op_name'}),
            **regular_op('fake_output1', {'type': None, 'kind': 'op', 'op': 'FakeOutput', 'name': 'my_output_name1'}),
            **regular_op('fake_output2', {'type': None, 'kind': 'op', 'op': 'FakeOutput', 'name': 'my_output_name2'}),

            **const('const1', int64_array(0)),
            **const('const2', int64_array(0)),
            **regular_op('add1', {'type': None, 'kind': 'op', 'op': 'Add', 'name': 'my_output_name1'}),
            **regular_op('add2', {'type': None, 'kind': 'op', 'op': 'Add', 'name': 'my_output_name2'}),
            **result('result1'),
            **result('result2'),
        }
        edges = [('input', 'some_op'),
                 ('some_op', 'fake_output1'),
                 ('some_op', 'fake_output2'),
                 ('fake_output1', 'result1'),
                 ('fake_output2', 'result2'),
                 ]
        graph = build_graph(nodes, edges)

        graph.graph['layout'] = 'NCHW'
        graph.stage = 'front'

        edges_ref = [('input', 'some_op'),
                     ('some_op', 'add1'),
                     ('const1', 'add1'),
                     ('some_op', 'add2'),
                     ('const2', 'add2'),
                     ('add1', 'result1'),
                     ('add2', 'result2'),
                     ]

        graph_ref = build_graph(nodes, edges_ref)

        FakeOutputResolver().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result1')
        self.assertTrue(flag, resp)
