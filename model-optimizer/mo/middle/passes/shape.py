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

from extensions.back.InsertLayoutPropagationTransposes import is_input_data_in_correct_layout, \
    is_output_data_in_correct_layout
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import PermuteAttrs
from mo.utils.error import Error


def apply_nhwc_to_nchw_permutation(graph: Graph):
    # Add NHWC to NCHW permutation for all data nodes (only for nodes without permutation)
    if graph.graph['layout'] == 'NCHW':
        return

    for node in graph.get_data_nodes():
        if node.has_and_set('nchw_layout'):
            continue

        # Get NHWC to NCHW permutation for N dims, where N = len(node.shape)
        permutation = PermuteAttrs().get_nhwc_to_nchw_permutation(len(node.shape))

        # Check that data node already has permutation
        skip_permutation = False
        for in_node in node.in_nodes():
            edge_attrs = node.graph.get_edge_data(in_node.id, node.id)[0]
            if 'permutation' in edge_attrs:
                skip_permutation = True
        for out_node in node.out_nodes():
            edge_attrs = node.graph.get_edge_data(node.id, out_node.id)[0]
            if 'permutation' in edge_attrs:
                skip_permutation = True

        if skip_permutation:
            continue

        # Set permutation to all in/out edges
        for in_node in node.in_nodes():
            PermuteAttrs.set_permutation(in_node, node, permutation)

        for out_node in node.out_nodes():
            PermuteAttrs.set_permutation(node, out_node, permutation)


def merge_nodes_permutations(graph: Graph):
    # Iterate over all data nodes and check all permutations for similarity
    # In case of equal permutations, this permutation will be set as attribute for data node
    # otherwise exception will be raised
    for node in graph.nodes():
        node = Node(graph, node)
        if node.kind != 'data':
            continue

        permutations = []

        # Get all permutations from in edges
        for in_node in node.in_nodes():
            edge_attrs = node.graph.get_edge_data(in_node.id, node.id)[0]
            if 'permutation' in edge_attrs:
                permutations.append(edge_attrs['permutation'])

        # Get all permutations from out edges
        for out_node in node.out_nodes():
            edge_attrs = node.graph.get_edge_data(node.id, out_node.id)[0]
            if 'permutation' in edge_attrs:
                permutations.append(edge_attrs['permutation'])

        # Check that all permutations are equal
        final_permutations = []
        for p in permutations:
            if p is not None:
                final_permutations.append(p.perm)
            else:
                final_permutations.append(int64_array(np.arange(node.shape.size)))

        if len(final_permutations) == 0:
            continue

        if not all([np.array_equal(final_permutations[0], perm) for perm in final_permutations]):
            raise Error('Permutations requested for {} data node are not equal! List of permutations: {}'
                        ''.format(node.name, [p.perm for p in permutations]))

        assert not node.has_valid('permutation') or np.array_equal(node.permutation, permutations[0])
        node['permutation'] = permutations[0]


def permute_data_nodes_attrs(graph: Graph):
    # Iterate over all data nodes and apply permutation if exists
    for node in graph.get_data_nodes():
        if not node.has_valid('permutation'):
            continue

        if len(node.in_nodes()) != 0:  # there are data nodes without input operation node inside the tensor iterator
            edge_attrs = graph.get_edge_data(node.in_node(0).id, node.id)[0]
            if is_output_data_in_correct_layout(node.in_node(0), edge_attrs['out']):
                log.debug('Do not permute data node attrs for node "{}" output port "{}"'.format(node.in_node(0).id,
                                                                                                 edge_attrs['out']))
                continue

        # Apply permutation for shape and value if exists
        if len(node.permutation.perm) == 0:
            continue
        node.shape = np.array(node.shape)[node.permutation.perm]
        if node.has_valid('value'):
            assert len(node.value.shape) == len(node.permutation.perm), \
                'Node {} has shape {} and permutation {} that does not match. Their lengths should be equal' \
                ''.format(node.name, node.value.shape, node.permutation.perm)
            node.value = np.array(node.value.transpose(node.permutation.perm))


def permute_op_nodes_attrs(graph: Graph):
    for node in graph.get_op_nodes():
        if node.has_valid('permute_attrs') and not node.has_and_set('nchw_layout'):
            try:
                node.permute_attrs.permute_attrs(node)
            except Exception as e:
                raise Error('Can\'t permute attrs for node {}. Error message: {}'.format(node.id, e))


def permute_input_data(graph: Graph):
    if graph.graph['layout'] != 'NHWC':
        return
    for node in graph.get_op_nodes():
        input_permutations = [(in_port, edge_attrs['input_permutation']) for in_port, edge_attrs in
                              node.in_edges().items() if 'input_permutation' in edge_attrs]
        for in_port, input_perm in input_permutations:
            permutation, port_info = input_perm
            direction, port = port_info.split(':')
            port = int(port)
            port_to_check = node.in_port(port) if direction == 'input' else node.out_port(port)
            if not is_input_data_in_correct_layout(node, in_port) and len(port_to_check.data.get_shape()) >= 4:
                permutation(node, port_info, in_port)


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

