# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.tf.NonConstBeginStridedSliceReplacement import NonConstBeginStridedSliceReplacement
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const


class NonConstBeginStridedSliceReplacementTests(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            # nodes from original graph
            'input': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'index': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'add': {'type': 'Add', 'kind': 'op', 'op': 'Add'},
            **const('slice_size', int64_array(1)),
            'begin': {'type': 'Pack', 'kind': 'op', 'op': 'Pack'},
            **const('begin_1', int64_array(0)),
            **const('begin_3', int64_array(0)),
            'end': {'type': 'Pack', 'kind': 'op', 'op': 'Pack'},
            **const('end_1', int64_array(0)),
            **const('end_3', int64_array(0)),
            **const('step', int64_array([1, 1, 1])),
            'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice',
                              'begin_mask': int64_array([0, 1, 0]), 'end_mask': int64_array([0, 1, 0]),
                              'shrink_axis_mask': int64_array([0, 1, 0]), 'name': 'non_const_begin_strided_slice'},
            'result': {'type': 'Result', 'kind': 'op', 'op': 'Result'},

            # nodes from the reference graph
            'unsqueeze': {'type': 'Unsqueeze', 'kind': 'op', 'op': 'Unsqueeze'},
            **const('unsqueeze_axis', int64_array(0)),
            'gather': {'type': 'Gather', 'kind': 'op', 'op': 'Gather'},
            **const('gather_axis', int64_array(1)),
            'squeeze': {'type': 'Squeeze', 'kind': 'op', 'op': 'Squeeze'},
            **const('squeeze_axis', int64_array(1)),
        }

        graph = build_graph(nodes_attributes,
                            [('input', 'strided_slice', {'out': 0, 'in': 0}),
                             ('begin_1', 'begin', {'out': 0, 'in': 0}),
                             ('index', 'begin', {'out': 0, 'in': 1}),
                             ('begin_3', 'begin', {'out': 0, 'in': 2}),
                             ('begin', 'strided_slice', {'out': 0, 'in': 1}),
                             ('end_1', 'end', {'out': 0, 'in': 0}),
                             ('index', 'add', {'out': 0, 'in': 0}),
                             ('slice_size', 'add', {'out': 0, 'in': 1}),
                             ('add', 'end', {'out': 0, 'in': 1}),
                             ('end_3', 'end', {'out': 0, 'in': 2}),
                             ('end', 'strided_slice', {'out': 0, 'in': 2}),
                             ('step', 'strided_slice', {'out': 0, 'in': 3}),
                             ('strided_slice', 'result', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        NonConstBeginStridedSliceReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input', 'gather', {'out': 0, 'in': 0}),
                                 ('gather_axis', 'gather', {'out': 0, 'in': 2}),
                                 ('index', 'unsqueeze', {'out': 0, 'in': 0}),
                                 ('unsqueeze_axis', 'unsqueeze', {'out': 0, 'in': 1}),
                                 ('unsqueeze', 'gather', {'out': 0, 'in': 1}),
                                 ('gather', 'squeeze', {'out': 0, 'in': 0}),
                                 ('squeeze_axis', 'squeeze', {'out': 0, 'in': 1}),
                                 ('squeeze', 'result', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
        self.assertTrue(graph.node[graph.get_nodes_with_attributes(op='Squeeze')[0]]['name'] ==
                        'non_const_begin_strided_slice')

    def test2_not_applied_transform(self):
        # the transformation is not applied if begin and end are constant
        nodes_attributes = {
            # nodes from original graph
            'input': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'begin': {'type': 'Pack', 'kind': 'op', 'op': 'Pack'},
            **const('begin_1', int64_array(0)),
            **const('begin_2', int64_array(0)),
            **const('begin_3', int64_array(0)),
            'end': {'type': 'Pack', 'kind': 'op', 'op': 'Pack'},
            **const('end_1', int64_array(0)),
            **const('end_2', int64_array(3)),
            **const('end_3', int64_array(0)),
            **const('step', int64_array([1, 1, 1])),
            'strided_slice': {'type': 'StridedSlice', 'kind': 'op', 'op': 'StridedSlice',
                              'begin_mask': int64_array([0, 1, 0]), 'end_mask': int64_array([0, 1, 0]),
                              'shrink_axis_mask': int64_array([0, 1, 0]), 'name': 'non_const_begin_strided_slice'},
            'result': {'type': 'Result', 'kind': 'op', 'op': 'Result'},
        }

        graph = build_graph(nodes_attributes,
                            [('input', 'strided_slice', {'out': 0, 'in': 0}),
                             ('begin_1', 'begin', {'out': 0, 'in': 0}),
                             ('begin_2', 'begin', {'out': 0, 'in': 1}),
                             ('begin_3', 'begin', {'out': 0, 'in': 2}),
                             ('begin', 'strided_slice', {'out': 0, 'in': 1}),
                             ('end_1', 'end', {'out': 0, 'in': 0}),
                             ('end_2', 'end', {'out': 0, 'in': 1}),
                             ('end_3', 'end', {'out': 0, 'in': 2}),
                             ('end', 'strided_slice', {'out': 0, 'in': 2}),
                             ('step', 'strided_slice', {'out': 0, 'in': 3}),
                             ('strided_slice', 'result', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        NonConstBeginStridedSliceReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input', 'strided_slice', {'out': 0, 'in': 0}),
                                 ('begin_1', 'begin', {'out': 0, 'in': 0}),
                                 ('begin_2', 'begin', {'out': 0, 'in': 1}),
                                 ('begin_3', 'begin', {'out': 0, 'in': 2}),
                                 ('begin', 'strided_slice', {'out': 0, 'in': 1}),
                                 ('end_1', 'end', {'out': 0, 'in': 0}),
                                 ('end_2', 'end', {'out': 0, 'in': 1}),
                                 ('end_3', 'end', {'out': 0, 'in': 2}),
                                 ('end', 'strided_slice', {'out': 0, 'in': 2}),
                                 ('step', 'strided_slice', {'out': 0, 'in': 3}),
                                 ('strided_slice', 'result', {'out': 0, 'in': 0})],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'result', check_op_attrs=True)
        self.assertTrue(flag, resp)
