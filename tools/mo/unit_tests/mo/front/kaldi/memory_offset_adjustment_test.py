# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.kaldi.memory_offset_adjustment import MemoryOffsetAdjustment
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class MemoruOffsetAdjustmentTests(unittest.TestCase):

    def test_several_memory_concat(self):
        graph = build_graph({'in': {'kind': 'op', 'op': None},
                             'memory_2': {'kind': 'op', 'op': 'MemoryOffset', 't': 2},
                             'memory_1': {'kind': 'op', 'op': 'MemoryOffset', 't': 1},
                             'memory__3': {'kind': 'op', 'op': 'MemoryOffset', 't': -3},
                             'concat': {'kind': 'op', 'op': 'Concat'}},
                            [('in', 'memory_2', {'out': 0}), ('in', 'memory_1', {'out': 1}),
                             ('in', 'memory__3', {'out': 3}),
                             ('memory_2', 'concat', {'in': 0}),
                             ('memory_1', 'concat', {'in': 1}),
                             ('in', 'concat', {'in': 2, 'out': 2}),
                             ('memory__3', 'concat', {'in': 3})],
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        ref_graph = build_graph({'in': {'kind': 'op', 'op': None},
                                 'memory__5': {'kind': 'op', 'op': 'MemoryOffset', 't': -5},
                                 'memory__1': {'kind': 'op', 'op': 'MemoryOffset', 't': -1},
                                 'memory__2': {'kind': 'op', 'op': 'MemoryOffset', 't': -2},
                                 'concat': {'kind': 'op', 'op': 'Concat'}},
                                [('in', 'memory__5', {'out': 3}), ('in', 'memory__1', {'out': 1}),
                                 ('in', 'memory__2', {'out': 2}),
                                 ('in', 'concat', {'in': 0, 'out': 0}),
                                 ('memory__2', 'concat', {'in': 2}),
                                 ('memory__1', 'concat', {'in': 1}),
                                 ('memory__5', 'concat', {'in': 3})],
                                nodes_with_edges_only=True)

        MemoryOffsetAdjustment().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_memory_before_several_memory_concat(self):
        graph = build_graph({'in': {'kind': 'op', 'op': None},
                             'memory_2': {'kind': 'op', 'op': 'MemoryOffset', 't': 2},
                             'memory_1': {'kind': 'op', 'op': 'MemoryOffset', 't': 1},
                             'memory__3': {'kind': 'op', 'op': 'MemoryOffset', 't': -3},
                             'concat': {'kind': 'op', 'op': 'Concat'}},
                            [('in', 'memory_1', {'out': 0}), ('memory_1', 'memory_2', {'out': 0}),
                             ('memory_1', 'memory__3', {'out': 0}),
                             ('memory_2', 'concat', {'in': 1}),
                             ('memory__3', 'concat', {'in': 0}),
                             ('memory_1', 'concat', {'in': 2, 'out': 0}),
                             ('in', 'concat', {'in': 3, 'out': 1})],
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        ref_graph = build_graph({'in': {'kind': 'op', 'op': None},
                                 'memory__3': {'kind': 'op', 'op': 'MemoryOffset', 't': -3},
                                 'memory__6': {'kind': 'op', 'op': 'MemoryOffset', 't': -5},
                                 'memory__2': {'kind': 'op', 'op': 'MemoryOffset', 't': -2},
                                 'concat': {'kind': 'op', 'op': 'Concat'}},
                                [('in', 'memory__2', {'out': 0}), ('memory__2', 'concat', {"in": 1, 'out': 0}),
                                 ('memory__2', 'memory__6', {'out': 0}),
                                 ('memory__6', 'concat', {'in': 0}),
                                 ('memory__3', 'concat', {'in': 3}),
                                 ('memory__2', 'concat', {"in": 2, 'out': 0}),
                                 ('in', 'memory__3', {'out': 1})],
                                nodes_with_edges_only=True)

        MemoryOffsetAdjustment().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'concat', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test_memory_parallel_several_memory_concat(self):
        graph = build_graph({'in': {'kind': 'op', 'op': None},
                             'memory_3': {'kind': 'op', 'op': 'MemoryOffset', 't': 3},
                             'memory__1': {'kind': 'op', 'op': 'MemoryOffset', 't': -1},
                             'memory_5': {'kind': 'op', 'op': 'MemoryOffset', 't': 5},
                             'concat': {'kind': 'op', 'op': 'Concat'},
                             'concat_1': {'kind': 'op', 'op': 'Concat'},
                             'split': {'kind': 'op', 'op': 'Split'}},
                            [('in', 'split', {'out': 0}), ('in', 'memory_5', {'out': 0}),
                             ('split', 'memory_3', {'out': 0}),
                             ('split', 'memory__1', {'out': 1}),
                             ('memory_3', 'concat', {'in': 0}),
                             ('memory__1', 'concat', {'in': 1}),
                             ('concat', 'concat_1', {'in': 0, 'out': 0}),
                             ('memory_5', 'concat_1', {'in': 1, 'out': 0}),
                             ],
                            nodes_with_edges_only=True)
        graph.stage = 'front'

        ref_graph = build_graph({'in': {'kind': 'op', 'op': None},
                                 'memory__4': {'kind': 'op', 'op': 'MemoryOffset', 't': -4},
                                 'memory__2': {'kind': 'op', 'op': 'MemoryOffset', 't': -2},
                                 'concat': {'kind': 'op', 'op': 'Concat'},
                                 'concat_1': {'kind': 'op', 'op': 'Concat'},
                                 'split': {'kind': 'op', 'op': 'Split'},
                                 },
                                [('in', 'split', {'out': 0}), ('in', 'concat_1', {'in': 1, 'out': 0}),
                                 ('split', 'concat', {'out': 0, 'in': 0}), ('split', 'memory__4', {'out': 1}),
                                 ('memory__4', 'concat', {'in': 1}),
                                 ('concat', 'memory__2'),
                                 ('memory__2', 'concat_1', {'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        MemoryOffsetAdjustment().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph, 'concat_1', check_op_attrs=True)
        self.assertTrue(flag, resp)
