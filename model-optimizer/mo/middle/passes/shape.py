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


from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import PermuteAttrs
from mo.utils.error import Error



def reverse_input_channels(graph: Graph):
    """
    Searchers for all type=Input nodes with 4D output tensors,
    tracks tensors down through non-shape-changing ops to the first type=Convolution or other channel-dependent nodes
    and reverse input channels in convolution weights.
    """
    candidates = set()
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('type') and node.type == 'Parameter' and len(node.out_nodes()) == 1 and node.out_node(
                0).shape.size == 4:
            candidates.add(node)
    log.debug('reverse_input_channels found candidates: {}'.format([c.node for c in candidates]))
    # Track down to the first convolutions
    convolutions = set()
    flip_passthrough = set()
    while len(candidates) > 0:
        op_node = candidates.pop()
        assert (len(op_node.out_nodes()) == 1)
        tensor_node = op_node.out_node(0)
        for consumer in tensor_node.out_nodes():
            if (consumer.has_valid('type') and
                    consumer.type == 'Convolution' and
                    consumer.in_node(1).has_valid('input_channel_dim') and
                    consumer.in_node(1).has_valid('shape') and
                    consumer.in_node(1).shape[consumer.in_node(1).input_channel_dim] == 3 and
                    consumer.in_node(1).has_valid('value')):
                convolutions.add(consumer)
            else:
                # TODO Use more reliable way
                if len(consumer.out_nodes()) == 1 and np.all(consumer.out_node().shape == tensor_node.shape):
                    candidates.add(consumer)
                    if consumer.has_valid('type') and (
                            consumer.type == 'ScaleShift' or consumer.type == 'BatchNormalization'):
                        flip_passthrough.add(consumer)
                else:
                    log.debug('Stop searching of conv candidate for channel reversing at node {}'.format(consumer.id))

    if len(convolutions) == 0:
        log.error('Reverse input channels are not applied -- appropriate convolutions were not found')

    for node in flip_passthrough:
        log.debug("Applying flip for ScaleShift: {}".format(node.name))
        assert node.has_valid('type') and (node.type == 'ScaleShift' or node.type == 'BatchNormalization')
        blobs = [node.in_node(i) for i in range(1, len(node.in_nodes()))]
        for blob in blobs:
            assert blob.has_valid('value')
            non_one_dimensions = np.where(blob.shape != 1)[0]
            assert len(non_one_dimensions) == 1
            assert blob.shape[non_one_dimensions[0]] == 3
            blob.value = np.flip(blob.value, non_one_dimensions[0])

    for conv in convolutions:
        if conv.op == 'DepthwiseConv2dNative':
            log.debug('out nodes: {}'.format(conv.out_node()))
            bottoms = conv.out_node().out_nodes()
            if len(bottoms) == 1 and bottoms[0].op == 'FakeQuantize':
                bottoms = bottoms[0].out_node().out_nodes()
            log.debug('bottoms: {}'.format(bottoms))
            log.debug('assumed conv: name = {}, op = {}'.format(bottoms[0].name, bottoms[0].op))
            if len(bottoms) > 0 and bottoms[0].op == 'Conv2D':
                bottom_conv = bottoms[0]
                # Flipping input channel for DepthwiseConv2dNative along doesn't do complete thing
                # We also need to flip input channels for the next convolution in groups
                ngroups = conv.group
                log.debug('ngroups = {}'.format(ngroups))
                bottom_channel_dim = bottom_conv.channel_dims[0]
                log.debug('bottom_challen_dim = {}'.format(bottom_channel_dim))
                bottom_channels = bottom_conv.in_node(0).shape[bottom_channel_dim]
                log.debug('bottom_channels = {}'.format(bottom_channels))
                assert (bottom_channels % ngroups == 0)
                multiplier = int(bottom_channels / ngroups)
                log.debug('multiplier = {}'.format(multiplier))
                bottom_weights = bottom_conv.in_node(1)
                tmp_shape_for_reorder = list(bottom_weights.value.shape)
                src_shape = list(tmp_shape_for_reorder)
                log.debug('weights shape = {}'.format(tmp_shape_for_reorder))
                assert (tmp_shape_for_reorder[bottom_weights.input_channel_dim] == bottom_channels)
                tmp_shape_for_reorder[bottom_weights.input_channel_dim] = ngroups
                tmp_shape_for_reorder = tmp_shape_for_reorder + [multiplier]
                log.debug('tmp_shape_for_reorder = {}'.format(tmp_shape_for_reorder))
                # temporary change shape of weights to do reordering
                # bottom_weights.value.shape = tuple(tmp_shape_for_reorder)
                bottom_weights.value = np.flip(bottom_weights.value.reshape(tuple(tmp_shape_for_reorder)),
                                               bottom_weights.input_channel_dim)
                # change shape of weights back
                log.debug('back to shape = {}'.format(tuple(src_shape)))
                bottom_weights.value = bottom_weights.value.reshape(tuple(src_shape))
                log.debug('final shape of weights = {}'.format(bottom_weights.value.shape))
                log.debug('shape as attr = {}'.format(bottom_weights.shape))
            else:
                log.error(
                    'Reverse input channels are not applied: there is no Conv2D after DepthwiseConv2dNative to ' +
                    'complete the flip')

        conv.in_node(1).value = np.flip(conv.in_node(1).value, conv.in_node(1).input_channel_dim)
        conv.in_node(1).shape = int64_array(conv.in_node(1).value.shape)
        log.debug('Applied reversing input channels for weights of convolution {}'.format(conv.id))
        log.debug('Shape was (shape){}, (value.shape){}'.format(conv.in_node(1).shape, conv.in_node(1).value.shape))
        log.debug('Flipped dim: {}'.format(conv.in_node(1).input_channel_dim))

