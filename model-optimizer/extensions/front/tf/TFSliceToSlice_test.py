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
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const, connect_front

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
    **regular_op_with_empty_data('cast', {'op': 'Cast', 'type': 'Convert'}),
}


class SliceReplacerTest(unittest.TestCase):
    # test case when input goes besides from TFSlice to other nodes
    def test_slice_replacer_begin_with_2_inputs(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'tfslice'),
            *connect_front('begin:0', '1:tfslice'),
            *connect_front('begin:0', '0:john_doe'),
            *connect_front('size:0', '2:tfslice'),
            *connect_front('tfslice:0', 'output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        TFSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', 'slice'),
            *connect_front('begin:0', 'slice:1'),
            *connect_front('begin:0', 'john_doe:1'),

            *connect_front('begin:0', 'end_const:0'),
            *connect_front('size:0', 'end_const:1'),
            *connect_front('size:0', 'equal:0'),

            *connect_front('int32_max:0', 'select:1'),
            *connect_front('minus_one:0', 'equal:1'),

            *connect_front('equal:0', 'select:0'),

            *connect_front('end_const:0', 'cast:0'),
            *connect_front('cast:0', 'select:2'),
            *connect_front('select:0', 'slice:2'),

            *connect_front('slice:0', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_slice_replacer(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', 'tfslice'),
            *connect_front('begin:0', '1:tfslice'),
            *connect_front('size:0', '2:tfslice'),
            *connect_front('tfslice:0', 'output'),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        TFSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            *connect_front('input:0', 'slice'),
            *connect_front('begin:0', '1:slice'),
            *connect_front('begin:0', '0:end_const'),
            *connect_front('size:0', '1:end_const'),
            *connect_front('size:0', '0:equal'),
            *connect_front('int32_max:0', '1:select'),
            *connect_front('minus_one:0', '1:equal'),
            *connect_front('equal:0', '0:select'),
            *connect_front('end_const:0', '0:cast'),
            *connect_front('cast:0', '2:select'),
            *connect_front('select:0', '2:slice'),
            *connect_front('slice:0', 'output'),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
