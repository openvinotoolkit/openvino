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
from mo.front.extractor import update_attrs
from mo.graph.graph import Node, Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node, merge_data_nodes, graph_clean_up_tf
from mo.middle.passes.fusing.helpers import get_next_operation
from mo.middle.pattern_match import apply_pattern
from mo.ops.op import PermuteAttrs, Op
from mo.ops.permute import Permute
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def reshape_squeeze_transform(graph: Graph, match: dict):
    reshape = match['reshape']
    output = match['output']
    if output.shape is None:
        return  # cannot really do anything if shape is dynamic
    reshape['shape'] = output.shape
    reshape.op = 'Reshape'
    reshape['type'] = 'Reshape'
    if not reshape.has_valid('dim'):
        # do not override value 'dim' if it is set. It may contain specific values like -1 and 0
        reshape['dim'] = reshape.shape.copy()
    update_attrs(reshape, 'shape_attrs', 'dim')


def convert_squeeze(graph: Graph):
    apply_pattern(
        graph,
        nodes=[
            ('reshape', dict(kind='op', op='Squeeze')),
            ('output', dict(kind='data'))],
        edges=[('reshape', 'output')],
        action=reshape_squeeze_transform
    )


def convert_reshape(graph: Graph):
    apply_pattern(
        graph,
        nodes=[
            ('shape', dict(kind='data')),
            ('reshape', dict(kind='op', op='Reshape')),
            ('output', dict(kind='data'))],
        edges=[('shape', 'reshape', {'in': 1}), ('reshape', 'output')],
        action=reshape_squeeze_transform
    )


def can_repack_fully_connected_weights_nhwc_to_nchw(fc_node: Node):
    """
    Checks that it is possible to repack weights of the FullyConnected layer if the Reshape layer is the input of the
    FullyConnected and satisfies several conditions.
    :param fc_node: the FullyConnected node to check
    :return: the result of the check
    """
    if len(fc_node.in_node(0).in_nodes()) != 1:
        return False

    reshape_node = fc_node.in_node(0).in_node(0)
    if not reshape_node.has_valid('type') or reshape_node.type != 'Reshape':
        return False

    if not reshape_node.in_node(0).has_valid('shape') or not reshape_node.out_node().has_valid('shape'):
        return False

    orig_shape = reshape_node.in_node(0).shape
    new_shape = reshape_node.out_node().shape

    # TODO a bit conservative condition; relax it checking specific dimensions that are involved in
    # NHWC to NCWH translation
    if len(orig_shape) == len(new_shape) and all(orig_shape == new_shape):
        return False

    # TODO here is a couple of limitations that makes this pass simpler; consider to relax them
    if len(orig_shape) == 4 and len(new_shape) == 2 and orig_shape[0] == new_shape[0]:
        # that means orig_shape is in NCHW and new_shape is in NC
        # and we need to map CHW part to C after HWC to CHW transform
        # Assuming that FullyConnected weights haven't been converted from IO to OI yet.
        # So format is IO.
        return True
    else:
        log.warning("Cannot do the complete NHWC to NCHW translation for FullyConnected weights. "
                    "The final model can be broken.")
        return False


def repack_fully_connected_weights_nhwc_to_nchw(graph: Graph):
    """
    Repack weights of FullyConnected layer as a part of nhwc_to_nchw translation if Reshape of
    that involves dimensions that we are repacking appears right before FullyConnected layer.
    """
    for node_id in graph.get_nodes_with_attributes(type='FullyConnected'):
        fc_node = Node(graph, node_id)

        if not can_repack_fully_connected_weights_nhwc_to_nchw(fc_node):
            continue

        reshape_node = fc_node.in_node(0).in_node(0)

        orig_shape = reshape_node.in_node(0).shape
        new_shape = reshape_node.out_node().shape

        # OK, here we are; need to repack fc_node.in_node(1) to maintain it compatible with original input order

        assert all(orig_shape != -1), 'Input shape for {} can not be negative.'.format(fc_node.id)
        assert all(new_shape != -1), 'Output shape for {} can not be negative.'.format(fc_node.id)
        assert orig_shape[1] * orig_shape[2] * orig_shape[3] == new_shape[1], \
            'Input shape does not correspond to output shape for layer {}.'.format(fc_node.id)
        assert fc_node.in_node(1).has_valid('value'), 'Node {} does not have value.'.format(fc_node.id)

        weights = fc_node.in_node(1)

        log.debug("orig_shape = {}".format(orig_shape))
        log.debug("new_shape = {}".format(new_shape))
        log.debug("weights.shape = {}".format(weights.shape))
        log.debug("weights.shape[1] = {}, new_shape[1] = {}".format(weights.shape[1], new_shape[1]))

        assert weights.shape[0] == new_shape[1], \
            'First dim of weights does not correspond to output shape of {}'.format(fc_node.id)
        # interpret I dimension of the weights as packed HWC
        # orig shape is already converted to NCHW, so provide transposed order for I repacking
        tmp_shape = (orig_shape[2], orig_shape[3], orig_shape[1], weights.shape[1])
        weights.value = np.transpose(weights.value.reshape(tmp_shape), (2, 0, 1, 3)).reshape(weights.shape)


def apply_nhwc_to_nchw_permutation(graph: Graph):
    # Add NHWC to NCHW permutation for all data nodes (only for nodes without permutation)
    if graph.graph['layout'] == 'NCHW':
        return
    for node in graph.nodes():
        node = Node(graph, node)
        if node.kind == 'data':
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
                final_permutations.append(np.arange(node.shape.size))

        if len(final_permutations) == 0:
            continue

        if not all([np.array_equal(final_permutations[0], perm) for perm in final_permutations]):
            raise Error(
                'Permutations requested for {} data node are not equal! List of permutations: {}'.format(node.name,
                                                                                                         [p.perm for
                                                                                                          p in
                                                                                                          permutations]))

        assert not node.has_valid('permutation') or np.array_equal(node.permutation, permutations[0])
        node['permutation'] = permutations[0]
        if node.permutation is not None and node.permutation.perm.size == 0:
            node.permutation = None


def permute_data_nodes_attrs(graph: Graph):
    # Iterate over all data nodes and apply permutation if exists
    for node in graph.nodes():
        node = Node(graph, node)
        if node.kind != 'data' or not node.has_valid('permutation'):
            continue

        # Apply permutation for shape and value if exists
        node.shape = np.array(node.shape)[node.permutation.perm]
        if node.has_valid('value'):
            if len(node.value.shape) != len(node.permutation.perm):
                log.warning('Node {} has shape {} and permutation {} that is not satisfied'.format(node.name, node.value.shape, node.permutation.perm))
                continue
            #print(node.name, node.value.shape, node.shape, node.permutation)
            node.value = np.array(node.value.transpose(node.permutation.perm))


def permute_op_nodes_attrs(graph: Graph):
    for node in graph.nodes():
        node = Node(graph, node)
        if node.kind == 'op' and node.has_valid('permute_attrs'):
            try:
                node.permute_attrs.permute_attrs(node)
            except Exception as e:
                raise Error('Can\'t permute attrs for node {}. Error message: {}'.format(node.id, e))


def reverse_input_channels(graph: Graph):
    """
    Searchers for all type=Input nodes with 4D output tensors,
    tracks tensors down through non-shape-changing ops to the first type=Convolution or other channel-dependent nodes
    and reverse input channels in convolution weights.
    """
    candidates = set()
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('type') and node.type == 'Input' and len(node.out_nodes()) == 1 and node.out_node(
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
            if len(bottoms) == 1 and bottoms[0].op == 'FakeQuantWithMinMaxVars':
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


def conv_flatten_concat_action(graph: Graph, match: dict):
    assert graph.graph['layout'] == 'NHWC'
    reshape_node = match['reshape']
    reshape_data_node = match['reshape_data']
    conv_name = match['conv'].name
    conv_data_node = match['conv_data']
    # the pattern should be applied only in case when the reshape operation changes number of dimensions
    if len(reshape_data_node.shape) == len(conv_data_node.shape) or reshape_node.has_and_set('nchw_layout'):
        return

    if len(reshape_data_node.out_nodes()) == 1 and reshape_data_node.out_node().has_valid('type') and \
        reshape_data_node.out_node().type == 'FullyConnected' and \
            can_repack_fully_connected_weights_nhwc_to_nchw(reshape_data_node.out_node()):
        log.info('There is a FullyConnected layer after the node "{}" which weights will be repacked. So there is no '
                 'need to insert Permute'.format(reshape_node.soft_get('name')))
        return
    graph.remove_edge(conv_data_node.id, reshape_node.id)

    permutation_order = PermuteAttrs.get_nchw_to_nhwc_permutation(len(conv_data_node.shape)).perm
    new_permute_op = Permute(graph, {'order': permutation_order})
    permute_data_node = new_permute_op.create_node_with_data([conv_data_node], dict(name=conv_name + '/Permute_'))
    graph.create_edge(permute_data_node, reshape_node)
    # Disable permutation for Reshape and Concat layers attributes
    PermuteAttrs.set_permutation(reshape_node, reshape_data_node, None)
    reshape_node['nchw_layout'] = True


def conv_flatten_concat(graph: Graph):
    apply_pattern(
        graph,
        nodes=[
            ('conv', dict(kind='op', type='Convolution')),
            ('conv_data', dict(kind='data')),
            ('reshape', dict(kind='op', type='Reshape')),
            ('reshape_data', dict(kind='data')),
        ],
        edges=[
            ('conv', 'conv_data'),
            ('conv_data', 'reshape'),
            ('reshape', 'reshape_data'),
        ],
        action=conv_flatten_concat_action
    )

    apply_pattern(
        graph,
        nodes=[
            ('real_conv', dict(kind='op', type='Convolution')),
            ('real_conv_data', dict(kind='data')),
            ('conv', dict(kind='op', type='ReLU')),
            ('conv_data', dict(kind='data')),
            ('reshape', dict(kind='op', type='Reshape')),
            ('reshape_data', dict(kind='data')),
        ],
        edges=[
            ('real_conv', 'real_conv_data'),
            ('real_conv_data', 'conv'),
            ('conv', 'conv_data'),
            ('conv_data', 'reshape'),
            ('reshape', 'reshape_data'),
        ],
        action=conv_flatten_concat_action
    )


def fuse_sequence_of_reshapes(graph: Graph):
    for node in list(graph.nodes()):
        if not graph.has_node(node):
            # data node can be already removed
            continue
        node = Node(graph, node)
        if (
                node.has_valid('type') and node.type == 'Reshape' and
                len(node.out_nodes()) == 1 and node.out_node().has_valid('kind') and node.out_node().kind == 'data' and
                len(node.out_node().out_nodes()) == 1):

            log.debug('First phase for Reshape: {}'.format(node.name))

            next_op = node.out_node().out_node()
            log.debug('second node: {}'.format(next_op.graph.node[next_op.id]))
            if next_op.has_valid('type') and next_op.type == 'Reshape':
                # Detected Reshape1 --> data --> Reshape2 pattern without side edges
                # Remove Reshape1
                log.debug('Second phase for Reshape: {}'.format(node.name))
                remove_op_node_with_data_node(graph, node)

    reshape_nodes = graph.get_op_nodes(op='Reshape')
    for reshape_node in reshape_nodes:
        in_ports = [port for port in reshape_node.in_ports().values() if not port.disconnected()]
        assert len(in_ports) in [1, 2], "`Reshape` node must have 2 inputs or 1 input with `dim`"
        if len(in_ports) == 2:
            previous_dim_op = reshape_node.in_port(1).get_source().node.op
            if previous_dim_op != 'Const':
                continue
            dim = reshape_node.in_port(1).get_connection().data.get_value()
        else:
            assert reshape_node.has_valid('dim'), "`Reshape` node with 1 input must have `dim` attribute"
            dim = reshape_node.dim

        in_shape = reshape_node.in_port(0).get_connection().data.get_shape()

        if np.array_equal(dim, in_shape) and len(reshape_node.out_nodes()):
            log.debug("Useless reshape with dim {} was deleted: {}".format(str(dim), reshape_node.name))
            reshape_node.out_port(0).get_connection().set_source(reshape_node.in_port(0).get_source())
