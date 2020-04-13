"""
 Copyright (c) 2018-2019 Intel Corporation

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

import numpy as np

from extensions.ops.gather import Gather
from extensions.ops.parameter import Parameter
from extensions.ops.sparse_weighted_sum import ExperimentalSparseWeightedSum
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.concat import Concat


class EmbeddingBagResolver(MiddleReplacementPattern):
    '''
     Replace EmbeddingBag with Gather or SparseWeightedSum.
     If shape of offsets is equal to shape of indices it means that offsets are obsolete because they have to define
     "bags" of shape 1 and we can remove offsets and replace EmbeddingBag with just Gather. In another case offsets must
      be used and EmbeddingBag can be replaced by SparseWeightedSum, but offsets must be pre-processed.
    '''
    enabled = True
    force_clean_up = True

    def find_and_replace_pattern(self, graph: Graph):
        weighted_sum_nodes = list()
        index_shape = None
        merge_offsets = True

        for node in graph.get_op_nodes(op='EmbeddingBag'):
            weights_shape = node.in_port(0).data.get_shape()
            indices_shape = node.in_port(1).data.get_shape()
            offsets_shape = node.in_port(2).data.get_shape()

            assert node.scale_grad_by_freq == 0

            if indices_shape[0] == offsets_shape[0]:
                # The simple case when we can replace EmbeddingBag with just Gather and not use offsets node at all
                gather = create_op_with_const_inputs(graph, Gather, {2: int64_array(0)},
                                                     {'name': node.name + '/Emb_Bag/Gather_'})

                node.in_port(0).get_connection().set_destination(gather.in_port(0))
                node.in_port(1).get_connection().set_destination(gather.in_port(1))
                node.out_port(0).get_connection().set_source(gather.out_port(0))
            else:
                assert node.mode == 0

                dense_shape = int64_array([offsets_shape[0], indices_shape[0]])
                default_index = int64_array(weights_shape[0])
                sweightedsum = create_op_with_const_inputs(graph, ExperimentalSparseWeightedSum,
                                                           {2: dense_shape, 4: default_index},
                                                           {'name': node.name + '/WeightedSum'})
                if index_shape is None:
                    index_shape = indices_shape[-1]
                else:
                    merge_offsets = merge_offsets and index_shape == indices_shape[-1]
                weighted_sum_nodes.append((sweightedsum, indices_shape[-1]))

                default_embeddings = np.zeros([1, weights_shape[-1]])
                weights_concat = create_op_with_const_inputs(graph, Concat, {1: default_embeddings},
                                                             {'axis': 0, 'in_ports_count': 2})
                node.in_port(0).get_connection().set_destination(weights_concat.in_port(0))
                node.in_port(1).get_connection().set_destination(sweightedsum.in_port(1))
                weights_concat.out_port(0).connect(sweightedsum.in_port(3))

                node.out_port(0).get_connection().set_source(sweightedsum.out_port(0))
        self.create_offsets_for_weighted_sum(graph, weighted_sum_nodes, merge_offsets, index_shape)

    def create_offsets_for_weighted_sum(self, graph, weighted_sum_nodes, merge_offsets, index_shape):
        new_offsets = None
        for i, (node, ind_shape) in enumerate(weighted_sum_nodes):
            if merge_offsets and len(weighted_sum_nodes) > 1:
                # generate single offsets input if possible
                if new_offsets is None:
                    shape = int64_array([len(weighted_sum_nodes), index_shape, 2])
                    new_offsets = Parameter(graph, {'name': 'Emb_Bag/offsets',
                                                    'shape': shape,
                                                    'data_type': np.int32}).create_node()
                    log.error(
                        'Pre-process of offsets is needed for generated input "Emb_Bag/offsets" of shape: {}. '
                        'Refer to the documentation on how to convert the ONNX* DLRM model'.format(shape),
                        extra={'is_warning': True})
                gather = create_op_with_const_inputs(graph, Gather, {1: int64_array(i), 2: int64_array(0)},
                                                     {'name': node.name + '/Gather_'})
                new_offsets.out_port(0).connect(gather.in_port(0))
                gather.out_port(0).connect(node.in_port(0))
            else:
                shape = int64_array([ind_shape, 2])
                new_offsets = Parameter(graph, {'name': 'Emb_Bag/offsets{}'.format(i),
                                                'shape': shape,
                                                'data_type': np.int32}).create_node()
                new_offsets.out_port(0).connect(node.in_port(0))
                log.error(
                    'Pre-process of offsets is needed for generated input "Emb_Bag/offsets{}" of shape: {}. '
                    'Refer to the documentation on how to convert the ONNX* DLRM model'.format(i, shape),
                    extra={'is_warning': True})
