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

from extensions.front.tf.slice_replacers import TFSliceToSliceReplacer
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, const

nodes = {
    **regular_op_with_empty_data('input', {'type': 'Parameter'}),
    **regular_op_with_empty_data('tfslice', {'op': 'TFSlice', 'type': None}),
    **const('begin', np.array(0)),
    **const('size', np.array([-1])),
    **result(),

    # nodes after replacement
    **const('minus_one', np.array(-1)),
    **const('int32_max', np.array(np.iinfo(np.int32).max)),
    **regular_op_with_empty_data('end_const', {'op': 'Add', 'type': 'Eltwise'}),
    **regular_op_with_empty_data('equal', {'op': 'Equal', 'type': 'Eltwise'}),
    **regular_op_with_empty_data('select', {'op': 'Select', 'type': 'Select'}),
    **regular_op_with_empty_data('slice', {'op': 'Slice', 'type': None}),
}


class SliceReplacerTest(unittest.TestCase):

    def test_slice_replacer(self):
        graph = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'tfslice', {'out': 0}),
            ('begin', 'tfslice', {'out': 0, 'in': 1}),
            ('size', 'tfslice', {'out': 0, 'in': 2}),
            ('tfslice', 'output', {'out': 0}),
        ], nodes_with_edges_only=True)
        graph.stage = 'front'

        TFSliceToSliceReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attrs=nodes, edges=[
            ('input', 'slice', {'out': 0}),
            ('begin', 'slice', {'out': 0, 'in': 1}),

            ('begin', 'end_const', {'out': 0, 'in': 0}),
            ('size', 'end_const', {'out': 0, 'in': 1}),
            ('size', 'equal', {'out': 0, 'in': 0}),

            ('int32_max', 'select', {'out': 0, 'in': 1}),
            ('minus_one', 'equal', {'out': 0, 'in': 1}),

            ('equal', 'select', {'out': 0, 'in': 0}),

            ('end_const', 'select', {'out': 0, 'in': 2}),
            ('select', 'slice', {'out': 0, 'in': 2}),

            ('slice', 'output', {'out': 0}),
        ], nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'output', check_op_attrs=True)
        self.assertTrue(flag, resp)
