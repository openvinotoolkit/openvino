# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.middle.RemoveDuplicationMemory import RemoveMemoryDuplicationPattern, MergeNeighborSplicePattern
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class RemoveMemoryDuplicationPatternTests(unittest.TestCase):

    def test_remove_duplication(self):
        graph = build_graph({'input': {'kind': 'op', 'op': 'Parameter'},
                             'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 6)},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 143]},
                             'placeholder_1': {'kind': 'op', 'op': None},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': range(-1, 2)},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             },
                            [('input', 'in_node'), ('in_node', 'splice_1'),
                             ('splice_1', 'splice_data_1'), ('splice_data_1', 'placeholder_1'),
                             ('in_node', 'splice_2'), ('splice_2', 'splice_data_2'), ('splice_data_2', 'placeholder_2'),
                             ],
                            nodes_with_edges_only=True)
        RemoveMemoryDuplicationPattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'input': {'kind': 'op', 'op': 'Parameter'},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 6)},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 143]},
                                 'placeholder_1': {'kind': 'op'},
                                 'crop_2': {'kind': 'op', 'op': 'Crop', 'offset': 52, 'dim': 39, 'axis': -1},
                                 'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                                 'placeholder_2': {'kind': 'op'},
                                 },
                                [
                                    ('input', 'in_node'), ('in_node', 'splice_1'),
                                    ('splice_1', 'splice_data_1'), ('splice_data_1', 'placeholder_1'),
                                    ('splice_data_1', 'crop_2'), ('crop_2', 'splice_data_2'),
                                    ('splice_data_2', 'placeholder_2'),
                                ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_2')
        self.assertTrue(flag, resp)

    def test_remove_duplication_with_crops(self):
        graph = build_graph({'input': {'kind': 'op', 'op': 'Parameter'},
                             'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 6)},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 143]},
                             'crop_1': {'kind': 'op', 'op': 'Crop', 'offset': 13, 'dim': 13, 'axis': -1},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': range(-1, 2)},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 39]},
                             'crop_2': {'kind': 'op', 'op': 'Crop', 'offset': 13, 'dim': 13, 'axis': -1},
                             },
                            [('input', 'in_node'), ('in_node', 'splice_1'),
                             ('splice_1', 'splice_data_1'), ('splice_data_1', 'crop_1'),
                             ('in_node', 'splice_2'), ('splice_2', 'splice_data_2'), ('splice_data_2', 'crop_2'),
                             ],
                            nodes_with_edges_only=True)
        RemoveMemoryDuplicationPattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'input': {'kind': 'op', 'op': 'Parameter'},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 6)},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 143]},
                                 'crop_1': {'kind': 'op', 'op': 'Crop', 'offset': 13, 'dim': 13},
                                 'crop_2': {'kind': 'op', 'op': 'Crop', 'offset': 65, 'dim': 13, 'axis': -1},
                                 },
                                [
                                    ('input', 'in_node'), ('in_node', 'splice_1'),
                                    ('splice_1', 'splice_data_1'),
                                    ('splice_data_1', 'crop_1'), ('splice_data_1', 'crop_2'),
                                ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'crop_2')
        self.assertTrue(flag, resp)

    def test_remove_duplication_neibor(self):
        graph = build_graph({'input': {'kind': 'op', 'op': 'Parameter'},
                             'in_node': {'kind': 'data', 'shape': [1, 13]},
                             'splice_1': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 1)},
                             'splice_data_1': {'kind': 'data', 'shape': [1, 78], 'value': None},
                             'placeholder_1': {'kind': 'op', 'op': None},
                             'splice_2': {'kind': 'op', 'op': 'Splice', 'context': range(0, 2)},
                             'splice_data_2': {'kind': 'data', 'shape': [1, 26], 'value': None},
                             'placeholder_2': {'kind': 'op', 'op': None},
                             },
                            [('input', 'in_node'), ('in_node', 'splice_1'),
                             ('splice_1', 'splice_data_1'), ('splice_data_1', 'placeholder_1'),
                             ('in_node', 'splice_2'), ('splice_2', 'splice_data_2'), ('splice_data_2', 'placeholder_2'),
                             ],
                            nodes_with_edges_only=True)
        MergeNeighborSplicePattern().find_and_replace_pattern(graph)
        ref_graph = build_graph({'input': {'kind': 'op', 'op': 'Parameter'},
                                 'in_node': {'kind': 'data', 'shape': [1, 13]},
                                 'splice_1': {'kind': 'op', 'op': 'Splice', 'context': range(-5, 2)},
                                 'splice_data_1': {'kind': 'data', 'shape': [1, 91], 'value': None},
                                 'crop_1': {'kind': 'op', 'op': 'Crop', 'offset': 0, 'dim': 78, 'axis': -1},
                                 'crop_1_data': {'kind': 'data', 'shape': [1, 78]},
                                 'placeholder_1': {'kind': 'op'},
                                 'crop_2': {'kind': 'op', 'op': 'Crop', 'offset': 65, 'dim': 26, 'axis': -1},
                                 'splice_data_2': {'kind': 'data', 'shape': [1, 26], 'value': None},
                                 'placeholder_2': {'kind': 'op'},
                                 },
                                [
                                    ('input', 'in_node'), ('in_node', 'splice_1'),
                                    ('splice_1', 'splice_data_1'), ('splice_data_1', 'crop_1'),
                                    ('crop_1', 'crop_1_data'), ('crop_1_data', 'placeholder_1'),
                                    ('splice_data_1', 'crop_2'), ('crop_2', 'splice_data_2'),
                                    ('splice_data_2', 'placeholder_2'),
                                ],
                                nodes_with_edges_only=True
                                )

        (flag, resp) = compare_graphs(graph, ref_graph, 'placeholder_2')
        self.assertTrue(flag, resp)
