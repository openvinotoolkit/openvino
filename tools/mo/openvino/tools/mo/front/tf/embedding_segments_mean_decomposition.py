# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.ConvertLike import ConvertLike
from openvino.tools.mo.ops.ReduceOps import ReduceSum
from openvino.tools.mo.ops.broadcast import Broadcast
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.elementwise import Div, Equal
from openvino.tools.mo.ops.embedding_bag import EmbeddingSegmentsSum
from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.ops.select import Select
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class EmbeddingSegmentsMeanDecomposition(FrontReplacementPattern):
    """
    This transformation decomposes EmbeddingSegmentsMean operation into EmbeddingSegmentSum operations taking into
    account that summed up embedding vectors for each vector must be normalized appropriately by a coefficient
    equal to a number of gathered embedding vectors for each object. If there is no gathered embedding vector
    for an object, the coefficient equals one.

    Approximate computation scheme (Cast operations omitted) for the normalization coefficients:

                                                                          Const(0)
    segment_ids -> Unsqueeze(axis=1) -----------------\                     |
                                                       \                   \/
                                                        ---> Equal() --> Select --> ReduceSum(axis=0) --> Norm. Coeff.
                                                       /                   /\
    Range(0, num_segments) -> Unsqueeze(axis=0)------ /                    |
                                                                        Const(1)
    """
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.tf.embedding_segments_operation_fusing import \
            EmbeddingSegmentsOperationMultipleFeaturesFusing, EmbeddingSegmentsOperationSingleFeatureFusing
        return [EmbeddingSegmentsOperationMultipleFeaturesFusing, EmbeddingSegmentsOperationSingleFeatureFusing]

    def find_and_replace_pattern(self, graph: Graph):
        for embedding_segments_mean in graph.get_op_nodes(op='EmbeddingSegmentsMean'):
            embedding_segments_mean_name = embedding_segments_mean.soft_get('name',
                                                                            embedding_segments_mean.id)
            embedding_table_input = embedding_segments_mean.in_port(0)
            segment_ids_input = embedding_segments_mean.in_port(2)
            num_segments_input = embedding_segments_mean.in_port(3)

            # TODO: support EmbeddingSegmentsMean with specified weights vector.
            # now this case has not appeared in models so far so EmbeddingSegmentsOperation fusion
            # transformations do not handle it either
            if embedding_segments_mean.is_in_port_connected(5):
                return

            # 1. compute indices membership matrix, i.e. which indices belong to some object
            # the shape of this matrix is [num_segments, num_indices]
            non_norm_range_1_to_num_segments = create_op_with_const_inputs(graph, Range,
                                                                           {0: int64_array(0),
                                                                            2: int64_array(1)},
                                                                           {'name': embedding_segments_mean_name +
                                                                                    '/Range1ToNumSegments',
                                                                            'output_type': np.int64})
            num_segments_input.get_connection().add_destination(non_norm_range_1_to_num_segments.in_port(1))

            range_1_to_num_segments = ConvertLike(graph, {'name': embedding_segments_mean_name +
                                                                  '/Range1ToNumSegmentsNorm'}
                                                  ).create_node()
            range_1_to_num_segments.in_port(0).connect(non_norm_range_1_to_num_segments.out_port(0))
            num_segments_input.get_connection().add_destination(range_1_to_num_segments.in_port(1))

            unsqueeze_range_1_to_num_segments = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(1)},
                                                                            {'name': embedding_segments_mean_name +
                                                                                     '/Range1ToNumSegmentsUnsqueeze'})
            unsqueeze_range_1_to_num_segments.in_port(0).connect(range_1_to_num_segments.out_port(0))
            unsqueeze_segment_ids = create_op_with_const_inputs(graph, Unsqueeze, {1: int64_array(0)},
                                                                {'name': embedding_segments_mean_name +
                                                                         '/SegmentIdsUnsqueeze'})
            segment_ids_input.get_connection().add_destination(unsqueeze_segment_ids.in_port(0))
            boolean_membership_matrix = Equal(graph, {'name': embedding_segments_mean_name +
                                                              '/BooleanMembershipMatrix'}
                                              ).create_node()
            boolean_membership_matrix.in_port(0).connect(unsqueeze_range_1_to_num_segments.out_port(0))
            boolean_membership_matrix.in_port(1).connect(unsqueeze_segment_ids.out_port(0))
            shape_of_membership_matrix = Shape(graph, {'name': embedding_segments_mean_name +
                                                               '/ShapeOfMembershipMatrix'}
                                               ).create_node([boolean_membership_matrix])
            one_scalar_constant = Const(graph, {'name': embedding_segments_mean_name + '/OneScalar',
                                                'value': int64_array([1])}).create_node()
            one_constant = Broadcast(graph, {'name': embedding_segments_mean_name + '/One'}
                                     ).create_node([one_scalar_constant,
                                                    shape_of_membership_matrix])
            zero_constant = Const(graph, {'name': embedding_segments_mean_name + '/Zero',
                                          'value': int64_array(0)}).create_node()
            membership_matrix = Select(graph, {'name': embedding_segments_mean_name + '/MembershipMatrix',
                                               'auto_broadcast': 'numpy'}).create_node([boolean_membership_matrix,
                                                                                        one_constant,
                                                                                        zero_constant])

            # 2. compute a number of indices belong to each object from the batch
            # it computes the normalization coefficients
            num_indices_per_object = create_op_with_const_inputs(graph, ReduceSum,
                                                                 {1: int64_array(1)},
                                                                 {'name': embedding_segments_mean_name +
                                                                          '/NumIndicesPerObject'})
            num_indices_per_object.in_port(0).connect(membership_matrix.out_port(0))

            # 3. replace zero coefficient (zero number of indices belong to an object) with one
            # because for such object the single default embedding vector is used
            where_zero_number = Equal(graph, {'name': embedding_segments_mean_name +
                                                      '/WhereZeroIndicesNumber'}
                                      ).create_node([num_indices_per_object, zero_constant])
            normalized_num_indices_per_object = Select(graph, {'name': embedding_segments_mean_name +
                                                                       '/NormNumIndicesPerObject',
                                                               'auto_broadcast': 'numpy'}
                                                       ).create_node([where_zero_number,
                                                                      one_scalar_constant,
                                                                      num_indices_per_object])

            # 4. cast normalized_num_indices_per_object to the same type as embedding vector table
            norm_coefficients = ConvertLike(graph, {'name': embedding_segments_mean_name +
                                                            '/NormCoefficients'}
                                            ).create_node()
            norm_coefficients.in_port(0).connect(normalized_num_indices_per_object.out_port(0))
            embedding_table_input.get_connection().add_destination(norm_coefficients.in_port(1))

            # 5. replace EmbeddingSegmentMean with EmbeddingSegmentSum
            embedding_segments_sum = EmbeddingSegmentsSum(graph, {'name': embedding_segments_mean_name +
                                                                          '/EmbeddingSegmentsSum'}
                                                          ).create_node()
            for in_port in embedding_segments_mean.in_ports():
                if embedding_segments_mean.is_in_port_connected(in_port):
                    embedding_segments_mean.in_port(in_port).get_connection().set_destination(
                        embedding_segments_sum.in_port(in_port))

            # 6. normalize EmbeddingSegmentSum results by computed coefficients
            result_node = Div(graph, {'name': embedding_segments_mean_name +
                                              '/Div'}
                              ).create_node([embedding_segments_sum, norm_coefficients])
            embedding_segments_mean.out_port(0).get_connection().set_source(result_node.out_port(0))

            rename_nodes([(embedding_segments_mean, embedding_segments_mean_name + '/AbandonedName'),
                          (result_node, embedding_segments_mean_name)])
            graph.remove_nodes_from([embedding_segments_mean.id])
