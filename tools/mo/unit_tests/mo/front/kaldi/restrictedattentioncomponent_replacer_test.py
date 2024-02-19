# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.front.kaldi.restrictedattentioncomponent_replacer \
    import RestrictedAttentionComponentReplacer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph, regular_op, connect_front, const


class RestrictedAttentionComponentReplacerTest(unittest.TestCase):
    nodes = {
        **regular_op('placeholder', {}),
        **regular_op('restrictedattention', {'op': 'restrictedattentioncomponent'}),
        **regular_op('placeholder_out', {}),
        **regular_op('reshape_1', {'type': 'Reshape'}),
        **const('reshape_1_shape', int64_array([10, -1])),
        **regular_op('split_1', {'kind': 'op', 'type': 'VariadicSplit'}),
        **const('split_1_axis', int64_array(1)),
        **const('split_1_shape', int64_array([44, 55, 47])),
        **regular_op('memoryoffset_1_0', {'type': None,
                                          't': -2, 'has_default': False}),
        **regular_op('memoryoffset_1_1', {'type': None,
                                          't': 2, 'has_default': False}),
        **regular_op('concat_1', {'type': 'Concat'}),
        **regular_op('split_2', {'type': 'VariadicSplit'}),
        **const('split_2_axis', int64_array(1)),
        **const('split_2_shape', int64_array([44, 3])),
        **regular_op('einsum_1', {'type': 'Einsum', 'equation': 'ij,ik->i'}),
        **regular_op('reshape_helper_1', {'type': 'Reshape'}),
        **const('reshape_helper_1_shape', int64_array([10, 1])),
        **regular_op('mul', {'type': 'Multiply'}),
        **const('mul_scale', mo_array(0.5, dtype=float)),
        **regular_op('add', {'type': 'Add'}),
        **regular_op('softmax', {'type': 'SoftMax'}),
        **regular_op('reshape_helper_3', {'type': 'Reshape'}),
        **const('reshape_helper_3_shape', int64_array([10, 1, 3])),
        **regular_op('memoryoffset_2_0', {'type': None,
                                          't': -2, 'has_default': False}),
        **regular_op('memoryoffset_2_1', {'type': None,
                                          't': 2, 'has_default': False}),
        **regular_op('concat_2', {'type': 'Concat'}),
        **regular_op('reshape_helper_2', {'type': 'Reshape'}),
        **const('reshape_helper_2_shape', int64_array([10, 55, 3])),
        **regular_op('einsum_2', {'type': 'Einsum', 'equation': 'ijk,ilk->ij'}),
        **regular_op('concat_3', {'type': 'Concat'}),
        **regular_op('reshape_2', {'type': 'Reshape'}),
        **const('reshape_2_shape', int64_array([1, -1])),
    }

    def test_restrictedattentioncomponent(self):
        """
        Test case that validates if supgraph replaced by RestrictedAttentionComponentReplacer
        class instead of RestrictedAttention operator is correct.
        """
        graph = build_graph(self.nodes, [
            *connect_front('placeholder', '0:restrictedattention'),
            *connect_front('restrictedattention', 'placeholder_out')
        ], nodes_with_edges_only=True)

        graph.stage = 'front'

        restricted_attention_node = graph.nodes['restrictedattention']
        restricted_attention_node['num_left_inputs'] = 1
        restricted_attention_node['num_right_inputs'] = 1
        restricted_attention_node['num_heads'] = 10
        restricted_attention_node['key_dim'] = 44
        restricted_attention_node['value_dim'] = 55
        restricted_attention_node['time_stride'] = 2
        restricted_attention_node['key_scale'] = 0.5

        ref_graph = build_graph(self.nodes, [
            ('placeholder', 'reshape_1', {'in': 0, 'out': 0}),
            ('reshape_1_shape', 'reshape_1', {'in': 1, 'out': 0}),
            ('reshape_1', 'split_1', {'in': 0, 'out': 0}),
            ('split_1_axis', 'split_1', {'in': 1, 'out': 0}),
            ('split_1_shape', 'split_1', {'in': 2, 'out': 0}),
            ('split_1', 'memoryoffset_1_0', {'in': 0, 'out': 0}),
            ('split_1', 'memoryoffset_1_1',  {"in": 0, 'out': 0}),
            ('split_1', 'memoryoffset_2_0',  {"in": 0, 'out': 1}),
            ('split_1', 'memoryoffset_2_1',  {"in": 0, 'out': 1}),
            ('split_1', 'split_2', {'in': 0, 'out': 2}),
            ('split_2_axis', 'split_2', {'in': 1, 'out': 0}),
            ('split_2_shape', 'split_2', {'in': 2, 'out': 0}),
            ('memoryoffset_1_0', 'concat_1', {'in': 0, 'out': 0}),
            ('split_1', 'concat_1', {'in': 1, 'out': 0}),
            ('memoryoffset_1_1', 'concat_1', {'in': 2, 'out': 0}),
            ('concat_1', 'einsum_1', {'in': 0, 'out': 0}),
            ('split_2', 'einsum_1', {'in': 1, 'out': 0}),
            ('einsum_1', 'reshape_helper_1', {'in': 0, 'out': 0}),
            ('reshape_helper_1_shape', 'reshape_helper_1', {'in': 1, 'out': 0}),
            ('reshape_helper_1', 'mul', {'in': 0, 'out': 0}),
            ('mul_scale', 'mul', {'in': 1, 'out': 0}),
            ('mul', 'add', {'in': 1, 'out': 0}),
            ('split_2', 'add', {'in': 0, 'out': 1}),
            ('add', 'softmax', {'in': 0, 'out': 0}),
            ('memoryoffset_2_0', 'concat_2', {'in': 0, 'out': 0}),
            ('split_1', 'concat_2', {'in': 1, 'out': 1}),
            ('memoryoffset_2_1', 'concat_2', {'in': 2, 'out': 0}),
            ('concat_2', 'reshape_helper_2', {'in': 0, 'out': 0}),
            ('reshape_helper_2_shape', 'reshape_helper_2', {'in': 1, 'out': 0}),
            ('reshape_helper_2', 'einsum_2', {'in': 0, 'out': 0}),
            ('softmax', 'reshape_helper_3', {'in': 0, 'out': 0}),
            ('reshape_helper_3_shape', 'reshape_helper_3', {'in': 1, 'out': 0}),
            ('reshape_helper_3', 'einsum_2', {'in': 1, 'out': 0}),
            ('einsum_2', 'concat_3', {'in': 0, 'out': 0}),
            ('softmax', 'concat_3', {'in': 1, 'out': 0}),
            ('concat_3', 'reshape_2', {'in': 0, 'out': 0}),
            ('reshape_2_shape', 'reshape_2', {'in': 1, 'out': 0}),
            ('reshape_2', 'placeholder_out')
        ], nodes_with_edges_only=True)

        RestrictedAttentionComponentReplacer().find_and_replace_pattern(graph)

        (flag, resp) = compare_graphs(graph, ref_graph,
                                      'placeholder_out', check_op_attrs=True)
        self.assertTrue(flag, resp)
