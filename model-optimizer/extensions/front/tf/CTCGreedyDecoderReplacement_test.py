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

import numpy as np
import unittest

from extensions.front.tf.CTCGreedyDecoderReplacement import CTCGreedyDecoderReplacement, CTCGreedyDecoderReplacement2
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph, const


class CTCGreedyDecoderReplacementTests(unittest.TestCase):
    def test1(self):
        nodes_attributes = {
            # nodes from original graph
            'logits': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'seq_len': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'decoder': {'kind': 'op', 'op': 'CTCGreedyDecoder'},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},
            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},

            # new nodes
            'new_decoder': {'kind': 'op', 'op': 'CTCGreedyDecoder', 'use_mask_format': True},
            **const('squeeze_axes', int64_array([2, 3])),
            'squeeze_dec_seq': {'kind': 'op', 'op': 'Squeeze'},
            'cast_to_int': {'kind': 'op', 'op': 'Cast'},
        }

        graph = build_graph(nodes_attributes,
                            [('logits', 'decoder', {'out': 0, 'in': 0}),
                             ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('decoder', 'cast', {'out': 1, 'in': 0}),
                             ('cast', 'sparse_to_dense', {'out': 0}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderReplacement().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('logits', 'decoder', {'out': 0, 'in': 0}),
                                 ('seq_len', 'decoder', {'out': 0, 'in': 1}),
                                 ('decoder', 'squeeze_dec_seq', {'out': 0, 'in': 0}),
                                 ('squeeze_axes', 'squeeze_dec_seq', {'out': 0, 'in': 1}),
                                 ('squeeze_dec_seq', 'cast_to_int', {'out': 0, 'in': 0}),
                                 ('cast_to_int', 'last', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertEqual(len(graph.get_op_nodes(op='Cast')) == 1 and
                         graph.get_op_nodes(op='Cast')[0]['name'] == 'sparse_to_dense', True,
                         'Name is not inherited from original node for CTCGreedyDecoderReplacement')
        self.assertTrue(flag, resp)

    def test2(self):
        nodes_attributes = {
            # nodes from original graph
            'logits': {'type': 'Parameter', 'kind': 'op', 'op': 'Parameter'},
            'transpose': {'kind': 'op', 'op': 'Transpose'},
            'shape': {'kind': 'op', 'op': 'ShapeOf'},
            'shape_1': {'kind': 'op', 'op': 'ShapeOf'},
            'strided_slice': {'kind': 'op', 'op': 'StridedSlice'},
            **const('stack', int64_array([1])),
            **const('stack1', int64_array([2])),
            **const('stack2', int64_array([1])),
            'strided_slice_1': {'kind': 'op', 'op': 'StridedSlice'},
            **const('stack_1', int64_array([0])),
            **const('stack1_1', int64_array([1])),
            **const('stack2_1', int64_array([1])),
            'dims': {'kind': 'op', 'op': 'Pack'},
            'fill': {'kind': 'op', 'op': 'Fill'},
            'decoder': {'kind': 'op', 'op': 'CTCGreedyDecoder'},
            'cast': {'kind': 'op', 'op': 'Cast'},
            'sparse_to_dense': {'kind': 'op', 'op': 'SparseToDense'},

            # new nodes
            **const('unsqueeze_batch_size_axis', int64_array(0)),
            'unsqueeze_batch_size': {'kind': 'op', 'op': 'Unsqueeze'},
            **const('unsqueeze_time_size_axis', int64_array(0)),
            'unsqueeze_time_size': {'kind': 'op', 'op': 'Unsqueeze'},
            'seq_mask_shape': {'kind': 'op', 'op': 'Concat'},
            'sequence_mask': {'kind': 'op', 'op': 'Broadcast'},
            **const('one', np.array([1.0], dtype=np.float)),
            **const('squeeze_axes', int64_array([2, 3])),
            'squeeze_dec_seq': {'kind': 'op', 'op': 'Squeeze'},
            'cast_to_int': {'kind': 'op', 'op': 'Cast'},

            'last': {'type': None, 'value': None, 'kind': 'op', 'op': 'Result'},
        }

        graph = build_graph(nodes_attributes,
                            [('logits', 'transpose', {'out': 0}),
                             ('transpose', 'shape', {'out': 0}),
                             ('transpose', 'shape_1', {'out': 0}),
                             ('transpose', 'decoder', {'out': 0, 'in': 0}),
                             ('shape', 'strided_slice', {'out': 0, 'in': 0}),
                             ('stack', 'strided_slice', {'out': 0, 'in': 1}),
                             ('stack1', 'strided_slice', {'out': 0, 'in': 2}),
                             ('stack2', 'strided_slice', {'out': 0, 'in': 3}),
                             ('shape_1', 'strided_slice_1', {'out': 0, 'in': 0}),
                             ('stack_1', 'strided_slice_1', {'out': 0, 'in': 1}),
                             ('stack1_1', 'strided_slice_1', {'out': 0, 'in': 2}),
                             ('stack2_1', 'strided_slice_1', {'out': 0, 'in': 3}),
                             ('strided_slice', 'dims', {'out': 0, 'in': 0}),
                             ('dims', 'fill', {'out': 0, 'in': 0}),
                             ('strided_slice_1', 'fill', {'out': 0, 'in': 1}),
                             ('fill', 'decoder', {'out': 0, 'in': 1}),
                             ('decoder', 'sparse_to_dense', {'out': 0, 'in': 0}),
                             ('decoder', 'cast', {'out': 1, 'in': 0}),
                             ('cast', 'sparse_to_dense', {'out': 0}),
                             ('sparse_to_dense', 'last', {'out': 0, 'in': 0}),
                             ], nodes_with_edges_only=True)
        graph.stage = 'front'
        CTCGreedyDecoderReplacement2().find_and_replace_pattern(graph)

        graph_ref = build_graph(nodes_attributes,
                                [('logits', 'transpose', {'out': 0}),
                                 ('transpose', 'shape', {'out': 0}),
                                 ('transpose', 'shape_1', {'out': 0}),
                                 ('transpose', 'decoder', {'out': 0, 'in': 0}),
                                 ('shape', 'strided_slice', {'out': 0, 'in': 0}),
                                 ('stack', 'strided_slice', {'out': 0, 'in': 1}),
                                 ('stack1', 'strided_slice', {'out': 0, 'in': 2}),
                                 ('stack2', 'strided_slice', {'out': 0, 'in': 3}),
                                 ('shape_1', 'strided_slice_1', {'out': 0, 'in': 0}),
                                 ('stack_1', 'strided_slice_1', {'out': 0, 'in': 1}),
                                 ('stack1_1', 'strided_slice_1', {'out': 0, 'in': 2}),
                                 ('stack2_1', 'strided_slice_1', {'out': 0, 'in': 3}),
                                 ('strided_slice', 'unsqueeze_batch_size', {'out': 0, 'in': 0}),
                                 ('unsqueeze_batch_size_axis', 'unsqueeze_batch_size', {'out': 0, 'in': 1}),
                                 ('strided_slice_1', 'unsqueeze_time_size', {'out': 0, 'in': 0}),
                                 ('unsqueeze_time_size_axis', 'unsqueeze_time_size', {'out': 0, 'in': 1}),
                                 ('unsqueeze_batch_size', 'seq_mask_shape', {'out': 0, 'in': 1}),
                                 ('unsqueeze_time_size', 'seq_mask_shape', {'out': 0, 'in': 0}),
                                 ('one', 'sequence_mask', {'out': 0, 'in': 0}),
                                 ('seq_mask_shape', 'sequence_mask', {'out': 0, 'in': 1}),
                                 ('sequence_mask', 'decoder', {'out': 0, 'in': 1}),
                                 ('decoder', 'squeeze_dec_seq', {'out': 0, 'in': 0}),
                                 ('squeeze_axes', 'squeeze_dec_seq', {'out': 0, 'in': 1}),
                                 ('squeeze_dec_seq', 'cast_to_int', {'out': 0, 'in': 0}),
                                 ('cast_to_int', 'last', {'out': 0, 'in': 0}),
                                 ],
                                nodes_with_edges_only=True)

        (flag, resp) = compare_graphs(graph, graph_ref, 'last', check_op_attrs=True)
        self.assertEqual(len(graph.get_op_nodes(op='Cast')) == 1 and
                         graph.get_op_nodes(op='Cast')[0]['name'] == 'sparse_to_dense', True,
                         'Name is not inherited from original node for CTCGreedyDecoderReplacement2')
        self.assertTrue(flag, resp)
