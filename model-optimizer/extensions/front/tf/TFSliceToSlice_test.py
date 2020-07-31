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

from extensions.front.tf.TFSliceToSlice import TFSliceToSliceReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const, connect

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('tfslice', {'op': 'TFSlice', 'type': None}),
    **const('begin', np.array(0)),
    **const('size', np.array([-1])),
    **regular_op_with_empty_data('john_doe', {'op': 'Sum', 'type': None}),
    **result(),

    # nodes after replacement
    **const('minus_one', np.array(-1)),
    **const('int32_max', np.array(np.iinfo(np.int32).max)),
    **regular_op_with_empty_data('end_const', {'op': 'Add', 'type': 'Add'}),
    **regular_op_with_empty_data('equal', {'op': 'Equal', 'type': 'Equal'}),
    **regular_op_with_empty_data('select', {'op': 'Select', 'type': 'Select'}),
    **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),
}


class SliceReplacerTest(unittest.TestCase):
    # test case when input goes besides from TFSlice to other nodes
    def test_slice_replacer_begin_with_2_inputs(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'tfslice'),
            *connect('begin:0', '1:tfslice', on_front=True),
            *connect('begin:0', '0:john_doe', on_front=True),
            *connect('size:0', '2:tfslice', on_front=True),
            *connect('tfslice:0', 'output', on_front=True),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        TFSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            *connect('input:0', 'slice', on_front=True),
            *connect('begin:0', 'slice:1', on_front=True),
            *connect('begin:0', 'john_doe:1', on_front=True),

            *connect('begin:0', 'end_const:0', on_front=True),
            *connect('size:0', 'end_const:1', on_front=True),
            *connect('size:0', 'equal:0', on_front=True),

            *connect('int32_max:0', 'select:1', on_front=True),
            *connect('minus_one:0', 'equal:1', on_front=True),

            *connect('equal:0', 'select:0', on_front=True),

            *connect('end_const:0', 'select:2', on_front=True),
            *connect('select:0', 'slice:2', on_front=True),

            *connect('slice:0', 'output', on_front=True),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_slice_replacer(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('input:0', 'tfslice', on_front=True),
            *connect('begin:0', '1:tfslice', on_front=True),
            *connect('size:0', '2:tfslice', on_front=True),
            *connect('tfslice:0', 'output', on_front=True),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        TFSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            *connect('input:0', 'slice', on_front=True),
            *connect('begin:0', '1:slice', on_front=True),
            *connect('begin:0', '0:end_const', on_front=True),
            *connect('size:0', '1:end_const', on_front=True),
            *connect('size:0', '0:equal', on_front=True),
            *connect('int32_max:0', '1:select', on_front=True),
            *connect('minus_one:0', '1:equal', on_front=True),
            *connect('equal:0', '0:select', on_front=True),
            *connect('end_const:0', '2:select', on_front=True),
            *connect('select:0', '2:slice', on_front=True),
            *connect('slice:0', 'output', on_front=True),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
