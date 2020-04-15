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

from extensions.front.tf.sparse_weighted_sum import ExperimentalSparseWeightedSumFrontReplacer, \
    ExperimentalSparseWeightedSumFrontReplacer2
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class ExperimentalSparseWeightedSumFrontReplacersTest(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            'input_indices': {'shape': int64_array([5, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_values': {'shape': int64_array([5]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_dense_shape': {'shape': int64_array([2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_params_table': {'shape': int64_array([10, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_default_value': {'shape': int64_array([]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},

            'identity_spw': {'kind': 'op', 'op': 'Identity'},
            'gather0_1': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'gather0_2': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'reshape0': {'kind': 'op', 'op': 'Reshape'},
            'where0': {'kind': 'op', 'op': 'Where'},
            'greaterequal0': {'kind': 'op', 'op': 'GreaterEqual'},
            'sparse_fill_empty_rows': {'kind': 'op', 'op': 'SparseFillEmptyRows'},
            'unique': {'kind': 'op', 'op': 'Unique'},
            'strided_slice': {'kind': 'op', 'op': 'StridedSlice'},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'gather': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'sparse_segment_sum': {'kind': 'op', 'op': 'SparseSegmentSum'},
            'reshape': {'kind': 'op', 'op': 'Reshape'},
            'tile': {'kind': 'op', 'op': 'Tile', 'type': 'Tile'},
            'select': {'kind': 'op', 'op': 'Select'},

            'sparse_weighted_sum': {'kind': 'op', 'op': 'ExperimentalSparseWeightedSum'},

            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        }

        graph = build_graph(nodes_attributes,
                            [('input_indices', 'gather0_1', {'out': 0, 'in': 0}),
                                ('input_dense_shape', 'identity_spw', {'out': 0, 'in': 0}),
                                ('input_values', 'greaterequal0', {'out': 0, 'in': 0}),
                                ('input_values', 'gather0_2', {'out': 0, 'in': 0}),
                                ('input_params_table', 'gather', {'out': 0, 'in': 0}),
                                ('input_default_value', 'sparse_fill_empty_rows', {'out': 0, 'in': 3}),
                                 
                                ('gather0_1', 'sparse_fill_empty_rows', {'out': 0, 'in': 0}),
                                ('gather0_2', 'sparse_fill_empty_rows', {'out': 0, 'in': 1}),
                                ('identity_spw', 'sparse_fill_empty_rows', {'out': 0, 'in': 2}),
                                ('reshape0', 'gather0_1', {'out': 0, 'in': 1}),
                                ('reshape0', 'gather0_2', {'out': 0, 'in': 1}),
                                ('where0', 'reshape0', {'out': 0, 'in': 0}),
                                ('greaterequal0', 'where0', {'out': 0, 'in': 0}),
                                ('sparse_fill_empty_rows', 'unique', {'out': 1, 'in': 0}),
                                ('sparse_fill_empty_rows', 'strided_slice', {'out': 0, 'in': 0}),
                                ('sparse_fill_empty_rows', 'reshape', {'out': 2, 'in': 0}),
                                ('unique', 'sparse_segment_sum', {'out': 1, 'in': 1}),
                                ('unique', 'gather', {'out': 0, 'in': 1}),
                                ('strided_slice', 'cast', {'out': 0, 'in': 0}),
                                ('gather', 'sparse_segment_sum', {'out': 0, 'in': 0}),
                                ('cast', 'sparse_segment_sum', {'out': 0, 'in': 2}),
                                ('sparse_segment_sum', 'select', {'out': 0, 'in': 2}),
                                ('reshape', 'tile', {'out': 0, 'in': 0}),
                                ('tile', 'select', {'out': 0, 'in': 0}),
                                ('select', 'last', {'out': 0, 'in': 0}),
                            ], nodes_with_edges_only=True)
        graph.stage = 'front'
        ExperimentalSparseWeightedSumFrontReplacer().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input_indices', 'sparse_weighted_sum', {'in': 0}),
                                    ('input_values', 'sparse_weighted_sum', {'in': 1}),
                                    ('input_dense_shape', 'sparse_weighted_sum', {'in': 2}),
                                    ('input_params_table', 'sparse_weighted_sum', {'in': 3}),
                                    ('input_default_value', 'sparse_weighted_sum', {'in': 4}),
                                    ('sparse_weighted_sum', 'last', {'in': 0}),],
                                    nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2(self):
        nodes_attributes = {
            'input_indices': {'shape': int64_array([5, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_values': {'shape': int64_array([5]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_dense_shape': {'shape': int64_array([2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_params_table': {'shape': int64_array([10, 3, 4]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_default_value': {'shape': int64_array([]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},

            'identity_spw': {'kind': 'op', 'op': 'Identity'},
            'gather0_1': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'gather0_2': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'reshape0': {'kind': 'op', 'op': 'Reshape'},
            'where0': {'kind': 'op', 'op': 'Where'},
            'greaterequal0': {'kind': 'op', 'op': 'GreaterEqual'},
            'sparse_fill_empty_rows': {'kind': 'op', 'op': 'SparseFillEmptyRows'},
            'unique': {'kind': 'op', 'op': 'Unique'},
            'strided_slice': {'kind': 'op', 'op': 'StridedSlice'},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'gather': {'kind': 'op', 'op': 'Gather', 'type': 'Gather'},
            'identity': {'kind': 'op', 'op': 'Identity'},
            'identity_1': {'kind': 'op', 'op': 'Identity'},
            'sparse_segment_sum': {'kind': 'op', 'op': 'SparseSegmentSum'},
            'reshape': {'kind': 'op', 'op': 'Reshape'},
            'tile': {'kind': 'op', 'op': 'Tile', 'type': 'Tile'},
            'select': {'kind': 'op', 'op': 'Select'},

            'sparse_weighted_sum': {'kind': 'op', 'op': 'ExperimentalSparseWeightedSum'},

            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        }

        graph = build_graph(nodes_attributes,
                            [('input_indices', 'gather0_1', {'out': 0, 'in': 0}),
                                ('input_dense_shape', 'identity_spw', {'out': 0, 'in': 0}),
                                ('input_values', 'greaterequal0', {'out': 0, 'in': 0}),
                                ('input_values', 'gather0_2', {'out': 0, 'in': 0}),
                                ('input_params_table', 'gather', {'out': 0, 'in': 0}),
                                ('input_default_value', 'sparse_fill_empty_rows', {'out': 0, 'in': 3}),
                                 
                                ('identity_spw', 'sparse_fill_empty_rows', {'out': 0, 'in': 2}),
                                ('gather0_1', 'sparse_fill_empty_rows', {'out': 0, 'in': 0}),
                                ('gather0_2', 'sparse_fill_empty_rows', {'out': 0, 'in': 1}),
                                ('reshape0', 'gather0_1', {'out': 0, 'in': 1}),
                                ('reshape0', 'gather0_2', {'out': 0, 'in': 1}),
                                ('where0', 'reshape0', {'out': 0, 'in': 0}),
                                ('greaterequal0', 'where0', {'out': 0, 'in': 0}),
                                ('sparse_fill_empty_rows', 'unique', {'out': 1, 'in': 0}),
                                ('sparse_fill_empty_rows', 'strided_slice', {'out': 0, 'in': 0}),
                                ('sparse_fill_empty_rows', 'reshape', {'out': 2, 'in': 0}),
                                ('unique', 'sparse_segment_sum', {'out': 1, 'in': 1}),
                                ('unique', 'gather', {'out': 0, 'in': 1}),
                                ('strided_slice', 'cast', {'out': 0, 'in': 0}),
                                ('gather', 'identity', {'out': 0, 'in': 0}),
                                ('identity', 'identity_1', {'out': 0, 'in': 0}),
                                ('identity_1', 'sparse_segment_sum', {'out': 0, 'in': 0}),
                                ('cast', 'sparse_segment_sum', {'out': 0, 'in': 2}),
                                ('sparse_segment_sum', 'select', {'out': 0, 'in': 2}),
                                ('reshape', 'tile', {'out': 0, 'in': 0}),
                                ('tile', 'select', {'out': 0, 'in': 0}),
                                ('select', 'last', {'out': 0, 'in': 0})],
                                nodes_with_edges_only=True)
        graph.stage = 'front'
        ExperimentalSparseWeightedSumFrontReplacer2().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input_indices', 'sparse_weighted_sum', {'in': 0}),
                                    ('input_values', 'sparse_weighted_sum', {'in': 1}),
                                    ('input_dense_shape', 'sparse_weighted_sum', {'in': 2}),
                                    ('input_params_table', 'sparse_weighted_sum', {'in': 3}),
                                    ('input_default_value', 'sparse_weighted_sum', {'in': 4}),
                                    ('sparse_weighted_sum', 'last', {'in': 0}),],
                                    nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
