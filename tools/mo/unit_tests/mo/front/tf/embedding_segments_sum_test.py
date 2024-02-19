# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.embedding_segments_operation_fusing import \
    EmbeddingSegmentsOperationMultipleFeaturesFusing, \
    EmbeddingSegmentsOperationSingleFeatureFusing
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, const


class EmbeddingSegmentsOperationFusingTest(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            'input_indices': {'shape': int64_array([5, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_values': {'shape': int64_array([5]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_dense_shape': {'shape': int64_array([2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_params_table': {'shape': int64_array([10, 3, 4]), 'type': 'Parameter', 'kind': 'op',
                                   'op': 'Parameter'},
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

            'split_for_indices': {'kind': 'op', 'op': 'Split'},
            'squeeze_for_indices': {'kind': 'op', 'op': 'Squeeze'},
            'split_for_dense_shape': {'kind': 'op', 'op': 'Split'},
            'squeeze_to_scalar': {'kind': 'op', 'op': 'Squeeze'},
            'cast_indices': {'kind': 'op', 'op': 'Cast'},
            'cast_segment_ids': {'kind': 'op', 'op': 'Cast'},
            'cast_default_value': {'kind': 'op', 'op': 'Cast'},
            'cast_number_segments': {'kind': 'op', 'op': 'Cast'},
            'embedding_segments_sum': {'kind': 'op', 'op': 'EmbeddingSegmentsSum'},

            **const('split_for_indices_axis', int64_array(1)),
            **const('split_for_dense_shape_axis', int64_array(0)),
            **const('squeeze_axis', int64_array([0])),
            **const('squeeze_for_indices_axis', int64_array([1])),

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
        EmbeddingSegmentsOperationSingleFeatureFusing().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input_indices', 'split_for_indices', {'in': 0}),
                                 ('split_for_indices_axis', 'split_for_indices', {'in': 1}),
                                 ('split_for_indices', 'squeeze_for_indices', {'in': 0}),
                                 ('squeeze_for_indices_axis', 'squeeze_for_indices', {'in': 1}),
                                 ('squeeze_for_indices', 'cast_segment_ids', {'in': 0}),
                                 ('cast_segment_ids', 'embedding_segments_sum', {'in': 2, 'out': 0}),
                                 ('input_values', 'cast_indices', {'in': 0}),
                                 ('cast_indices', 'embedding_segments_sum', {'in': 1}),
                                 ('input_dense_shape', 'split_for_dense_shape', {'in': 0}),
                                 ('split_for_dense_shape_axis', 'split_for_dense_shape', {'in': 1}),
                                 ('split_for_dense_shape', 'squeeze_to_scalar', {'in': 0}),
                                 ('squeeze_axis', 'squeeze_to_scalar', {'in': 1}),
                                 ('squeeze_to_scalar', 'cast_number_segments', {'in': 0}),
                                 ('cast_number_segments', 'embedding_segments_sum', {'in': 3, 'out': 0}),
                                 ('input_params_table', 'embedding_segments_sum', {'in': 0}),
                                 ('input_default_value', 'cast_default_value', {'in': 0}),
                                 ('cast_default_value', 'embedding_segments_sum', {'in': 4}),
                                 ('embedding_segments_sum', 'last', {'in': 0}), ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)

    def test2(self):
        nodes_attributes = {
            'input_indices': {'shape': int64_array([5, 2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_values': {'shape': int64_array([5]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_dense_shape': {'shape': int64_array([2]), 'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'input_params_table': {'shape': int64_array([10, 3, 4]), 'type': 'Parameter', 'kind': 'op',
                                   'op': 'Parameter'},
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
            'sparse_segment_mean': {'kind': 'op', 'op': 'SparseSegmentMean'},
            'reshape': {'kind': 'op', 'op': 'Reshape'},
            'tile': {'kind': 'op', 'op': 'Tile', 'type': 'Tile'},
            'select': {'kind': 'op', 'op': 'Select'},

            'split_for_indices': {'kind': 'op', 'op': 'Split'},
            'squeeze_for_indices': {'kind': 'op', 'op': 'Squeeze'},
            'split_for_dense_shape': {'kind': 'op', 'op': 'Split'},
            'squeeze_to_scalar': {'kind': 'op', 'op': 'Squeeze'},
            'cast_indices': {'kind': 'op', 'op': 'Cast'},
            'cast_segment_ids': {'kind': 'op', 'op': 'Cast'},
            'cast_default_value': {'kind': 'op', 'op': 'Cast'},
            'cast_number_segments': {'kind': 'op', 'op': 'Cast'},
            'embedding_segments_mean': {'kind': 'op', 'op': 'EmbeddingSegmentsMean'},

            **const('split_for_indices_axis', int64_array(1)),
            **const('split_for_dense_shape_axis', int64_array(0)),
            **const('squeeze_axis', int64_array([0])),
            **const('squeeze_for_indices_axis', int64_array([1])),

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
                             ('unique', 'sparse_segment_mean', {'out': 1, 'in': 1}),
                             ('unique', 'gather', {'out': 0, 'in': 1}),
                             ('strided_slice', 'cast', {'out': 0, 'in': 0}),
                             ('gather', 'identity', {'out': 0, 'in': 0}),
                             ('identity', 'identity_1', {'out': 0, 'in': 0}),
                             ('identity_1', 'sparse_segment_mean', {'out': 0, 'in': 0}),
                             ('cast', 'sparse_segment_mean', {'out': 0, 'in': 2}),
                             ('sparse_segment_mean', 'select', {'out': 0, 'in': 2}),
                             ('reshape', 'tile', {'out': 0, 'in': 0}),
                             ('tile', 'select', {'out': 0, 'in': 0}),
                             ('select', 'last', {'out': 0, 'in': 0})],
                            nodes_with_edges_only=True)
        graph.stage = 'front'
        EmbeddingSegmentsOperationMultipleFeaturesFusing().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('input_indices', 'split_for_indices', {'in': 0}),
                                 ('split_for_indices_axis', 'split_for_indices', {'in': 1}),
                                 ('split_for_indices', 'squeeze_for_indices', {'in': 0}),
                                 ('squeeze_for_indices_axis', 'squeeze_for_indices', {'in': 1}),
                                 ('squeeze_for_indices', 'cast_segment_ids', {'in': 0}),
                                 ('cast_segment_ids', 'embedding_segments_mean', {'in': 2, 'out': 0}),
                                 ('input_values', 'cast_indices', {'in': 0}),
                                 ('cast_indices', 'embedding_segments_mean', {'in': 1}),
                                 ('input_dense_shape', 'split_for_dense_shape', {'in': 0}),
                                 ('split_for_dense_shape_axis', 'split_for_dense_shape', {'in': 1}),
                                 ('split_for_dense_shape', 'squeeze_to_scalar', {'in': 0}),
                                 ('squeeze_axis', 'squeeze_to_scalar', {'in': 1}),
                                 ('squeeze_to_scalar', 'cast_number_segments', {'in': 0}),
                                 ('cast_number_segments', 'embedding_segments_mean', {'in': 3, 'out': 0}),
                                 ('input_params_table', 'embedding_segments_mean', {'in': 0}),
                                 ('input_default_value', 'cast_default_value', {'in': 0}),
                                 ('cast_default_value', 'embedding_segments_mean', {'in': 4}),
                                 ('embedding_segments_mean', 'last', {'in': 0}), ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertTrue(flag, resp)
