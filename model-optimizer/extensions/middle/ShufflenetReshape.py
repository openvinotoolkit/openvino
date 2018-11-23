"""
 Copyright (c) 2018 Intel Corporation

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

import networkx as nx
import numpy as np

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape


class FeatureShuffleReshape(MiddleReplacementPattern):
    """
    This pass finds patterns like in shufflenet topology (Reshape->Transpose->Reshape) and will change attributes for
    first Reshape and Transpose operations to preserve original semantics.
    """

    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('reshape1', dict(kind='op', type='Reshape')),
                ('reshape1_data', dict(kind='data')),
                ('transpose', dict(kind='op', type='Permute')),
                ('transpose_data', dict(kind='data')),
                ('reshape2', dict(kind='op', type='Reshape')),
                ('reshape2_data', dict(kind='data')),
            ],
            edges=[('reshape1', 'reshape1_data'),
                   ('reshape1_data', 'transpose'),
                   ('transpose', 'transpose_data'),
                   ('transpose_data', 'reshape2'),
                   ('reshape2', 'reshape2_data'),
                   ]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        reshape1 = match['reshape1']
        reshape2 = match['reshape2']
        transpose = match['transpose']

        # Check that Reshape->Transpose->Reshape shuffle only feature channel
        input_shape = np.array(reshape1.in_node(0).shape)
        reshape1_shape = np.array(reshape1.out_node().shape)
        output_shape = np.array(reshape2.out_node().shape)

        # Check that input shape is 4D
        if len(input_shape) != 4:
            log.warning('Can\'t convert Reshape->Transpose({})->Reshape sequence due to input shape should be 4D '
                        '(instead of {}D)'.format(transpose.name, len(input_shape)))
            return

        # Check that output shape the same as input shape
        if not np.prod(input_shape) == np.prod(output_shape):
            log.warning('Can\'t convert Reshape->Transpose({})->Reshape sequence due to output shape should be equal '
                        'to input shape: {} and {}'.format(transpose.name, input_shape, output_shape))
            return

        # Input shapes can be either NCHW or NHWC, so in case of channel split, feature channel can be splited as
        # follows in comments below
        # So feature_dims_split list contains possible dims responsible for feature dim
        if graph.graph['layout'] == 'NCHW':
            # NC1C2HW or NC1C2(H*W)
            feature_dim = 1
            spatial_dims = [2, 3]
            feature_dims_split = np.array([feature_dim, feature_dim + 1])
        else:
            # NHWC1C2 or N(H*W)C1C2 or (N*H*W)C1C2
            feature_dim = 3
            spatial_dims = [1, 2]
            feature_dims_split = np.array([len(reshape1_shape) - 2, len(reshape1_shape) - 1])

        # Check that feature_dims_split suits reshape layer shape
        for dim in feature_dims_split:
            if dim < 0 or dim >= len(reshape1_shape):
                log.warning('Can\'t convert Reshape({}:{})->Transpose->Reshape sequence. Can\'t detect feature shuffle.'
                            ''.format(reshape1.shape, reshape1_shape))
                return

        if not np.prod(np.delete(reshape1_shape, feature_dims_split)) == np.prod(np.delete(input_shape, feature_dim)):
            log.warning('Can\'t convert Reshape->Transpose->Reshape sequence. Can\'t detect feature shuffle. {} '
                        'should be equal to {}'.format(np.prod(np.delete(reshape1_shape, feature_dims_split)),
                                                       np.prod(np.delete(input_shape, feature_dim))))
            return

        # Check transpose order
        if not np.array_equal(feature_dims_split[::-1], transpose.order[feature_dims_split]):
            log.warning('Can\'t convert Reshape->Transpose({})->Reshape sequence. Transpose operation should witch '
                        'feature order (given order: {})'.format(transpose.name, transpose.order))
            return

        # Now we are sure that Reshape->Transpose->Reshape shuffle feature dims
        # So, then we change Reshape and Transpose attrs to suite NCHW layout

        # The resulting shape for Reshape1 layer : [N,C1,C2,(H*W)]
        new_reshape1_shape = np.concatenate((np.array([input_shape[0]]),
                                             np.array(reshape1_shape[feature_dims_split]),
                                             np.array([np.prod(input_shape[spatial_dims])])))

        new_transpose_order = np.array([0, 2, 1, 3])
        new_transpose_shape = np.array(new_reshape1_shape[new_transpose_order])

        reshape1.out_node().shape = new_reshape1_shape
        transpose.order = new_transpose_order
        transpose.out_node().shape = new_transpose_shape

        # Preserve layers from conversion to NCHW (in case of NHWC topology layout)
        reshape1['nchw_layout'] = True
        reshape1.out_node()['nchw_layout'] = True
        transpose['nchw_layout'] = True
        transpose.out_node()['nchw_layout'] = True


class ReshapeSoftmaxReshape(MiddleReplacementPattern):
    """
    In case of NHWC this pass finds patterns Reshape(-1,2)->Softmax and changes first Reshape dims for NCHW format.
    This transformation is necessary because after conversion to NCHW this sequence will have wrong interpretation
    """

    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('reshape1', dict(kind='op', type='Reshape')),
                ('reshape1_data', dict(kind='data')),
                ('softmax', dict(kind='op', type='SoftMax')),
                ('softmax_data', dict(kind='data')),
            ],
            edges=[('reshape1', 'reshape1_data'),
                   ('reshape1_data', 'softmax'),
                   ('softmax', 'softmax_data'),
                   ])

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        if graph.graph['layout'] != 'NHWC':
            return

        reshape1 = match['reshape1']
        softmax = match['softmax']

        # Check that Reshape->Softmax->Reshape shuffle only feature channel
        input_shape = np.array(reshape1.in_node(0).shape)
        reshape1_shape = np.array(reshape1.out_node().shape)

        # Check that input shape is 4D
        if len(input_shape) != 4:
            log.warning('Can\'t convert Reshape({})->Softmax->Reshape sequence due to input shape should be 4D '
                        '(instead of {}D {})'.format(reshape1.name, len(input_shape), input_shape))
            return

        if len(reshape1_shape) != 2:
            log.warning('This pass expect 2D output tensor for first Reshape {} layer (given shape: {})'
                        ''.format(reshape1.name, reshape1_shape))
            return

        # Define feature dim
        feature_dim = 3
        spatial_dims = [1, 2]

        # Skip transform in case if spatial dims in input shape are equal to [1,1]
        if np.array_equal(input_shape[spatial_dims], np.array([1, 1])):
            log.info('Skip this transformation due to spatial dims are [1,1]')
            return

        # Check that Reshape1 has out dims [-1, feature_dims]
        if not (reshape1_shape[-1] == input_shape[-1] and reshape1_shape[0] == np.prod(
                np.delete(input_shape, feature_dim))):
            log.warning('Output shape for Reshape operation should be [{},{}] instead of {}'.format(
                np.prod(np.delete(input_shape, feature_dim)), input_shape[-1], reshape1_shape))
            return

        # Now we are sure that Reshape->Softmax suits for this transformation

        # The resulting shape for Reshape1 layer : [N,C,(H*W)]
        new_reshape1_shape = np.concatenate((np.array([input_shape[0]]),
                                             np.array([reshape1_shape[-1]]),
                                             np.array([np.prod(input_shape[spatial_dims])])))

        old_shape = np.array(reshape1.out_node().shape)
        reshape1.out_node().shape = new_reshape1_shape
        softmax.out_node().shape = new_reshape1_shape

        # Preserve layers from conversion to NCHW (in case of NHWC topology layout)
        reshape1['nchw_layout'] = True
        reshape1.out_node()['nchw_layout'] = True
        softmax['nchw_layout'] = True
        softmax.out_node()['nchw_layout'] = True

        # Create final Reshape to keep original shape for softmax output
        softmax_out_data = softmax.out_node()
        next_operation = softmax_out_data.out_node()
        # Save edge attributes & remove edge
        edge_attrs = graph.get_edge_data(softmax_out_data.id, next_operation.id)[0]
        graph.remove_edge(softmax_out_data.id, next_operation.id)

        reshape_op = Reshape(graph, dict(name="Reshape_", dim=np.array(old_shape)))
        reshape_out_data = reshape_op.create_node_with_data(inputs=[softmax_out_data])
        graph.add_edges_from([(reshape_out_data.id, next_operation.id, edge_attrs)])
