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

import logging as log

from extensions.ops.embedding_bag import EmbeddingSegmentsSum
from extensions.ops.split import Split
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, rename_nodes
from mo.ops.squeeze import Squeeze


class EmbeddingSegmentsSumFrontReplacer(FrontReplacementSubgraph):
    """
    The transformation looks for pattern (sub-graph) that performs extraction of embedding vectors from the parameters table
    for object feature values and sum up these embedding vectors for every object.
    Such sub-graph is met in the Wide and Deep model in case of the SINGLE categorical feature.
    """
    enabled = True

    def pattern(self):
        log.debug('Enabled EmbeddingSegmentsSum replacement')
        return dict(
            nodes=[
                ('identity_spw', dict(op='Identity')),
                ('gather0_1', dict(type='Gather')),
                ('gather0_2', dict(type='Gather')),
                ('reshape0', dict(op='Reshape')),
                ('where0', dict(op='Where')),
                ('greaterequal0', dict(op='GreaterEqual')),
                ('sparse_fill_empty_rows', dict(op='SparseFillEmptyRows')),
                ('unique', dict(op='Unique')),
                ('strided_slice', dict(op='StridedSlice')),
                ('cast', dict(op='Cast')),
                ('gather', dict(type='Gather')),
                ('sparse_segment_sum', dict(op='SparseSegmentSum')),
                ('reshape', dict(op='Reshape')),
                ('tile', dict(type='Tile')),
                ('select', dict(op='Select'))
            ],
            edges=[
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
                ('gather', 'sparse_segment_sum', {'out': 0, 'in': 0}),
                ('cast', 'sparse_segment_sum', {'out': 0, 'in': 2}),
                ('sparse_segment_sum', 'select', {'out': 0, 'in': 2}),
                ('reshape', 'tile', {'out': 0, 'in': 0}),
                ('tile', 'select', {'out': 0, 'in': 0})
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        identity_spw = match['identity_spw']
        gather0_1 = match['gather0_1']
        gather0_2 = match['gather0_2']
        greaterequal0 = match['greaterequal0']
        sparse_fill_empty_rows = match['sparse_fill_empty_rows']

        gather = match['gather']
        select = match['select']
        output_node_name = select.soft_get('name', select.id)
        log.debug('Found EmbeddingSegmentsSum pattern after {} with name {}'.format(sparse_fill_empty_rows.op,
                                                                                    sparse_fill_empty_rows.name))

        split_for_indices = create_op_with_const_inputs(graph, Split, {1: int64_array(1)}, {'num_splits': 2})
        squeeze_for_indices = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([1])})
        split_for_dense_shape = create_op_with_const_inputs(graph, Split, {1: int64_array(0)}, {'num_splits': 2})
        squeeze_to_scalar = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([0])})
        embedding_segments_sum = EmbeddingSegmentsSum(graph, {'name': output_node_name}).create_node()
        rename_nodes([(select, output_node_name + '/AbandonedName'), (embedding_segments_sum, output_node_name)])

        # connect parameters table
        gather.in_port(0).get_connection().set_destination(embedding_segments_sum.in_port(0))
        # connect indices values
        greaterequal0.in_port(0).get_connection().set_destination(embedding_segments_sum.in_port(1))
        # split and connect segment ids
        gather0_1.in_port(0).get_connection().set_destination(split_for_indices.in_port(0))
        squeeze_for_indices.in_port(0).connect(split_for_indices.out_port(0))
        embedding_segments_sum.in_port(2).connect(squeeze_for_indices.out_port(0))
        # split and connect number of segments
        identity_spw.in_port(0).get_connection().set_destination(split_for_dense_shape.in_port(0))
        squeeze_to_scalar.in_port(0).connect(split_for_dense_shape.out_port(0))
        embedding_segments_sum.in_port(3).connect(squeeze_to_scalar.out_port(0))
        # connect default value
        sparse_fill_empty_rows.in_port(3).get_connection().set_destination(embedding_segments_sum.in_port(4))
        # no input port for per_sample_weight

        identity_spw.in_port(0).disconnect()
        gather0_1.in_port(0).disconnect()
        gather0_2.in_port(0).disconnect()
        greaterequal0.in_port(0).disconnect()
        sparse_fill_empty_rows.in_port(2).disconnect()
        gather.in_port(0).disconnect()

        select.out_port(0).get_connection().set_source(embedding_segments_sum.out_port(0))
        graph.remove_nodes_from([gather0_1.id, gather0_2.id, greaterequal0.id, sparse_fill_empty_rows.id, select.id])


class EmbeddingSegmentsSumFrontReplacer2(FrontReplacementSubgraph):
    """
    The transformation looks for pattern (sub-graph) that performs extraction of embedding vectors from the parameters table
    for object feature values and sum up these embedding vectors for every object.
    Such sub-graph is met in the Wide and Deep model in case of MULTIPLE categorical features.
    """
    enabled = True

    def pattern(self):
        log.debug('Enabled EmbeddingSegmentsSum2 replacement')
        return dict(
            nodes=[
                ('identity_spw', dict(op='Identity')),
                ('gather0_1', dict(type='Gather')),
                ('gather0_2', dict(type='Gather')),
                ('reshape0', dict(op='Reshape')),
                ('where0', dict(op='Where')),
                ('greaterequal0', dict(op='GreaterEqual')),
                ('sparse_fill_empty_rows', dict(op='SparseFillEmptyRows')),
                ('unique', dict(op='Unique')),
                ('strided_slice', dict(op='StridedSlice')),
                ('cast', dict(op='Cast')),
                ('gather', dict(type='Gather')),
                ('identity', dict(op='Identity')),
                ('identity_1', dict(op='Identity')),
                ('sparse_segment_sum', dict(op='SparseSegmentSum')),
                ('reshape', dict(op='Reshape')),
                ('tile', dict(type='Tile')),
                ('select', dict(op='Select'))
            ],
            edges=[
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
                ('tile', 'select', {'out': 0, 'in': 0})
            ])

    def replace_sub_graph(self, graph: Graph, match: dict):
        identity_spw = match['identity_spw']
        gather0_1 = match['gather0_1']
        gather0_2 = match['gather0_2']
        greaterequal0 = match['greaterequal0']
        sparse_fill_empty_rows = match['sparse_fill_empty_rows']
        gather = match['gather']
        select = match['select']
        output_node_name = select.soft_get('name', select.id)

        log.debug('Found EmbeddingSegmentsSum2 pattern after {} with name {}'.format(sparse_fill_empty_rows.op,
                                                                                     sparse_fill_empty_rows.name))

        split_for_indices = create_op_with_const_inputs(graph, Split, {1: int64_array(1)},
                                                        {'num_splits': 2,
                                                         'name': output_node_name + '/SplitForIndices'})
        squeeze_for_indices = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([1])})
        split_for_dense_shape = create_op_with_const_inputs(graph, Split, {1: int64_array(0)},
                                                            {'num_splits': 2,
                                                             'name': output_node_name + '/SplitForDenseShape'})
        squeeze_to_scalar = create_op_with_const_inputs(graph, Squeeze, {1: int64_array([0])})
        embedding_segments_sum = EmbeddingSegmentsSum(graph, {'name': output_node_name}).create_node()
        rename_nodes([(select, output_node_name + '/AbandonedName'), (embedding_segments_sum, output_node_name)])

        # connect parameters table
        gather.in_port(0).get_connection().set_destination(embedding_segments_sum.in_port(0))
        # connect indices values
        greaterequal0.in_port(0).get_connection().set_destination(embedding_segments_sum.in_port(1))
        # split and connect segment ids
        gather0_1.in_port(0).get_connection().set_destination(split_for_indices.in_port(0))
        squeeze_for_indices.in_port(0).connect(split_for_indices.out_port(0))
        embedding_segments_sum.in_port(2).connect(squeeze_for_indices.out_port(0))
        # split and connect number of segments
        identity_spw.in_port(0).get_connection().set_destination(split_for_dense_shape.in_port(0))
        squeeze_to_scalar.in_port(0).connect(split_for_dense_shape.out_port(0))
        embedding_segments_sum.in_port(3).connect(squeeze_to_scalar.out_port(0))
        # connect default value
        sparse_fill_empty_rows.in_port(3).get_connection().set_destination(embedding_segments_sum.in_port(4))
        # no input port for per_sample_weight

        identity_spw.in_port(0).disconnect()
        gather0_1.in_port(0).disconnect()
        gather0_2.in_port(0).disconnect()
        greaterequal0.in_port(0).disconnect()
        sparse_fill_empty_rows.in_port(2).disconnect()
        gather.in_port(0).disconnect()

        select.out_port(0).get_connection().set_source(embedding_segments_sum.out_port(0))
        graph.remove_nodes_from([gather0_1.id, gather0_2.id, greaterequal0.id, sparse_fill_empty_rows.id, select.id])
