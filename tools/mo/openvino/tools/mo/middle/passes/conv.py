# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.front.common.layout import get_batch_dim, get_features_dim
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import add_attrs_props
from openvino.tools.mo.front.extractor import update_ie_fields
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes.fusing.helpers import get_value_id, get_tensor_id
from openvino.tools.mo.middle.pattern_match import apply_pattern


def pad_op_transform(graph: Graph, match: dict):
    op = match['op']
    pad_op: Node = match['pad_op']

    # to keep reshape-ability if Pad receives pads_begin/pads_end from shape subgraph
    if pad_op.in_port(1).get_source().node.soft_get('can_be_fused') is False:
        return

    if pad_op.mode != 'constant':
        log.info('The pad node "{}" with pad mode "{}" cannot be fused.'.format(pad_op.soft_get('name'), pad_op.mode))
        return

    if op.type == 'Pooling' and op.pool_method == 'max':
        return

    if pad_op.mode == 'constant':
        fill_value = pad_op.in_port(3).data.get_value()
        if fill_value is None or fill_value != 0.0:
            log.info('The pad node "{}" with non-zero fill value cannot be fused.'.format(pad_op.soft_get('name')))
            return

    input_tensor_dims = len(match['pad_output'].shape)
    for in_port in [1, 2]:
        pads = pad_op.in_port(in_port).data.get_value()
        if pads[get_features_dim(op.graph.graph['layout'], input_tensor_dims)] != 0 or \
                pads[get_batch_dim(op.graph.graph['layout'], input_tensor_dims)] != 0:
            log.info('The pad node "{}" with padding over feature/batch dimension cannot be fused.'.format(
                pad_op.soft_get('name')))
            return

    op.pad += np.concatenate([pad_op.in_port(1).data.get_value().reshape([-1, 1]),
                              pad_op.in_port(2).data.get_value().reshape([-1, 1])], axis=1)
    op.pad_spatial_shape = op.pad[op.spatial_dims]
    op['auto_pad'] = None
    if op.type == 'Pooling':
        op['exclude_pad'] = False
    assert (graph[match['pad_output'].node][match['op'].node][0]['in'] == 0)

    match['op'].in_port(0).disconnect()
    pad_op.in_port(0).get_connection().add_destination(match['op'].in_port(0))


def fuse_pad(graph: Graph):
    for op_type in ['Convolution', 'Pooling', 'Deconvolution']:
        apply_pattern(
            graph,
            nodes=[
                ('pad_op', dict(kind='op', op='Pad')),
                ('pad_output', dict(kind='data')),
                ('op', dict(kind='op', type=op_type))],
            edges=[('pad_op', 'pad_output'),
                   ('pad_output', 'op', {'in': 0})],
            action=pad_op_transform
        )


def muladd_to_scaleshift_action(graph: Graph, match: dict):
    mul = match['mul']
    add = match['add']
    output = match['output']

    # Pass works correctly only in case when node have only 1 output
    if len(mul.out_port(0).get_destinations()) > 1:
        return

    if mul.soft_get('can_be_scaleshift') is False or add.soft_get('can_be_scaleshift') is False:
        return

    mul_weights_id = get_value_id(mul)
    mul_input_id = get_tensor_id(mul)
    add_weights_id = get_value_id(add)

    if mul_weights_id is None:
        log.debug("Mul->Add to ScaleShift: Mul {} has no weights".format(mul.name))
        return
    if mul_input_id is None:
        log.debug("Mul->Add to ScaleShift: Mul {} has no input".format(mul.name))
        return
    if add_weights_id is None:
        log.debug("Mul->Add to ScaleShift: Add {} has no weights".format(add.name))
        return

    input = mul.in_node(mul_input_id)
    weights = mul.in_node(mul_weights_id)
    bias = add.in_node(add_weights_id)

    # Transform values
    weights.value = np.squeeze(weights.value)
    weights.shape = int64_array(weights.value.shape)

    bias.value = np.squeeze(bias.value)
    bias.shape = int64_array(bias.value.shape)

    # Broadcast weights if they are scalar
    if weights.value.ndim == 0 and bias.value.ndim == 1:
        weights.value = np.full(bias.shape, weights.value.item(), dtype=weights.value.dtype)
        weights.shape = int64_array(weights.value.shape)

    if bias.shape != weights.shape:
        log.warning('Mul->Add to ScaleShift conversion stopped {} != {}'.format(weights.shape, bias.shape))
        return

    if bias.value.ndim != weights.value.ndim or bias.value.size != weights.value.size:
        log.debug("Skipping Mul->Add to ScaleShift conversion for nodes {}, {} because of different weights "
                  "and biases".format(mul.name, add.name))
        return

    if bias.value.size == 1 and weights.value.size == 1:
        log.debug("Skipping Mul->Add to ScaleShift conversion for nodes {}, {}. Will be converted to Power"
                  "".format(mul.name, add.name))
        return

    op_name = "ScaleShift"

    log.debug("Fusing Mul->Add to {}. Input nodes: {} and {}, bias.shape = {}, weights.shape = {}"
              "".format(op_name, mul.id, add.id, bias.shape, weights.shape))

    graph.remove_edge(input.node, mul.id)
    graph.remove_edge(weights.node, mul.id)
    graph.remove_edge(bias.node, add.id)
    graph.remove_edge(add.node, output.id)

    op_node = graph.unique_id(mul.name + '/Fused{}_'.format(op_name))

    graph.add_node(op_node, **add_attrs_props(dict(kind='op', type=op_name, name=op_node, op=op_name,
                                                   data_type=input.data_type)))
    scsh = Node(graph, op_node)
    scsh.add_input_port(0)
    scsh.add_input_port(1)
    scsh.add_input_port(2)
    scsh.add_output_port(0)

    update_ie_fields(graph.node[op_node])

    graph.add_edges_from([
        (input.node, op_node, {'in': 0}),
        (weights.node, op_node, {'in': 1, 'bin': 'weights'}),
        (bias.node, op_node, {'in': 2, 'bin': 'biases'}),
        (op_node, output.node, {'out': 0})
    ])

    return


def batch_norm_fuse_action(graph: Graph, match: dict):
    """
    Multiply convolution kernel by batch normalization coefficient and remove mul op.
    """
    if match['norm'].value is None or match['kernel'].value is None:
        # cannot fuse non-const normalization coefficients
        return
    if len(graph.out_edges(match['conv_output'].node)) > 1 or len(graph.out_edges(match['kernel'].node)) > 1:
        # we cannot modify original kernel or convolution, if they are used multiple times
        # TODO make a copy of conv and kernel instead of this check
        return
    match['kernel'].value = match['kernel'].value * match['norm'].value
    graph.remove_edge(match['conv_output'].node, match['mul'].node)
    graph.remove_edge(match['mul'].node, match['mul_output'].node)
    # graph.remove_node(match['mul'].node)  # if we remove a node, next iteration over isomorphisms gives an error
    graph.add_edge(match['conv'].node, match['mul_output'].node, out=0)


def batch_norm_fuse(graph: Graph):
    apply_pattern(
        graph,
        nodes=[
            ('kernel', dict(kind='data')),
            ('conv', dict(kind='op', op='Conv2D')),
            ('conv_output', dict(kind='data')),
            ('norm', dict(kind='data')),
            ('mul', dict(kind='op', op='Mul')),
            ('mul_output', dict(kind='data'))],
        edges=[
            ('kernel', 'conv', {'in': 1}),
            ('conv', 'conv_output'),
            ('conv_output', 'mul', {'in': 0}),  # TODO get rid of explicit input port number, mul is a commutative op
            ('norm', 'mul', {'in': 1}),  # TODO get rig of explicit input port number, mul is a commutative op
            ('mul', 'mul_output')],
        action=batch_norm_fuse_action
    )
    return graph
