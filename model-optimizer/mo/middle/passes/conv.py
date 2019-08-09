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

from mo.front.common.layout import get_batch_dim, get_features_dim
from mo.front.common.partial_infer.utils import assign_dims_to_weights
from mo.front.extractor import add_attrs_props
from mo.front.extractor import update_ie_fields
from mo.graph.graph import Node, Graph
from mo.graph.port import Port
from mo.middle.passes.fusing.helpers import get_value_id, get_tensor_id, get_tensor_in_port, get_value_in_port
from mo.middle.pattern_match import apply_pattern
from mo.ops.const import Const
from mo.ops.scale_shift import ScaleShiftOp


def pad_op_transform(graph: Graph, match: dict):
    op = match['op']
    pad_op = match['pad_op']
    input_data = pad_op.in_node(0)
    pads = pad_op.in_node(1).value if len(pad_op.in_nodes()) == 2 else pad_op.pads

    if pad_op.mode != 'constant':
        log.info('The pad node "{}" with pad mode "{}" cannot be fused.'.format(pad_op.soft_get('name'), pad_op.mode))
        return

    if pad_op.mode == 'constant' and pad_op.fill_value != 0.0:
        log.info('The pad node "{}" with non-zero fill value cannot be fused.'.format(pad_op.soft_get('name')))
        return

    input_tensor_dims = len(match['pad_output'].shape)
    if np.any(pads[get_features_dim(op.graph.graph['layout'], input_tensor_dims)] != 0) or \
            np.any(pads[get_batch_dim(op.graph.graph['layout'], input_tensor_dims)] != 0):
        log.info('The pad node "{}" with padding over feature/batch dimension cannot be fused.'.format(
            pad_op.soft_get('name')))
        return

    op.pad += pads
    op.pad_spatial_shape = op.pad[op.spatial_dims]
    op['auto_pad'] = None
    if op.type == 'Pooling':
        op['exclude_pad'] = False
    assert (graph[match['pad_output'].node][match['op'].node][0]['in'] == 0)
    edge_attrs = graph.get_edge_data(match['pad_output'].id, match['op'].id)[0]
    graph.remove_edge(match['pad_output'].id, match['op'].id)
    graph.add_edge(input_data.id, match['op'].id, **{'in': 0, **edge_attrs})


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


def convert_matmul_to_fully_connected(graph: Graph):
    apply_pattern(
        graph,
        nodes=[
            ('matmul', dict(kind='op', op='MatMul')),
            ('output', dict(kind='data'))],
        edges=[('matmul', 'output')],
        action=matmul_to_fully_connected_action
    )


def matmul_to_fully_connected_action(graph: Graph, match: dict):
    log.debug('fully_connected_matched')
    matmul = match['matmul']
    input = matmul.in_node(0)
    weights = matmul.in_node(1)
    weights_consumers = graph.out_edges(weights.node)
    log.debug("len(input.shape) = {}, len(weights.shape) = {}, len(weights_consumers) = {}".format(
        len(input.shape) if input.shape is not None else None, len(weights.shape) if not None else None,
        len(weights_consumers) if weights_consumers is not None else None))

    if not (weights.value is not None and
            input.shape is not None and
            len(input.shape) >= 2 and
            weights.shape is not None and
            len(weights.shape) == 2 and
            len(weights_consumers) >= 1):
        matmul['can_be_fused'] = False
        return

    matmul['out-size'] = matmul.out_node().shape[-1]
    if graph.graph['cmd_params'].generate_experimental_IR_V10:
        matmul['type'] = 'MatMul'
    else:
        matmul['type'] = 'FullyConnected'
        matmul.in_edge(1)['bin'] = 'weights'
    matmul['channel_dims'] = len(match['output'].shape) - 1
    matmul['bias_addable'] = True
    matmul['input_channel_dim'] = 0  # MatMul weights in IO
    matmul['output_channel_dim'] = 1
    matmul['layout'] = 'NHWC'

    assign_dims_to_weights(matmul.in_node(1), None, matmul.input_channel_dim, matmul.output_channel_dim, 2)
    # Do not transpose weights in this pass, it will be done as a separate pass


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
    weights.shape = np.array(weights.value.shape, dtype=np.int64)

    bias.value = np.squeeze(bias.value)
    bias.shape = np.array(bias.value.shape, dtype=np.int64)

    # Broadcast weights if they are scalar
    if weights.value.ndim == 0 and bias.value.ndim == 1:
        weights.value = np.full(bias.shape, weights.value.item())
        weights.shape = np.array(weights.value.shape, dtype=np.int64)

    if bias.shape != weights.shape:
        log.warning('Mul->Add to ScaleShift conversion stoped {} != {}'.format(weights.shape, bias.shape))
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

    graph.add_node(op_node, **add_attrs_props(dict(kind='op', precision="FP32", type=op_name, name=op_node,
                                                   op=op_name, data_type=input.data_type)))
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


def convert_muladd_to_scaleshift(graph: Graph):
    if hasattr(graph, 'graph') and 'cmd_params' in graph.graph and graph.graph['cmd_params'].generate_experimental_IR_V10:
        return
    # TODO nGraph remove BEGIN
    apply_pattern(
        graph,
        nodes=[
            ('input', dict(kind='data')),
            ('weights', dict(kind='data')),
            ('bias', dict(kind='data')),
            ('mout', dict(kind='data')),
            ('output', dict(kind='data')),
            ('mul', dict(kind='op', op='Mul')),
            ('add', dict(kind='op', op='Add'))
        ],
        edges=[
            ('weights', 'mul'),
            ('input', 'mul'),
            ('mul', 'mout'),
            ('mout', 'add'),
            ('bias', 'add'),
            ('add', 'output'),
        ],
        action=muladd_to_scaleshift_action
    )
    # TODO nGraph remove END


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


def convert_add_or_mul_to_scaleshift(graph: Graph):
    if graph.graph['cmd_params'].generate_experimental_IR_V10:
        return
    graph.strict_mode = False
    for node in graph.get_op_nodes():
        if node.soft_get('op') in ['Add', 'Mul'] and len(node.in_ports()) == 2:

            tensor_port, value_port = get_tensor_in_port(node), get_value_in_port(node)

            if tensor_port is not None and not tensor_port.disconnected() and value_port is not None and node.soft_get('can_be_scaleshift') is not False:
                original_value = value_port.data.get_value()
                if original_value.size == 1:
                    continue

                # Remove 1 dims from value array (should be 1D)
                value_port.data.set_value(np.squeeze(original_value))  # Updated shapes accordingly

                # Create ScaleShift operation
                scsh_op = ScaleShiftOp(graph, dict(name='ScaleShift/{}'.format(node.name))).create_node()

                if node.op == 'Mul':
                    # Create fake biases for scale shift node
                    const_op = Const(graph, dict(name='{}/biases'.format(scsh_op.name),
                                                 value=np.zeros(value_port.data.get_shape(), dtype=np.float32),
                                                 shape=np.array(value_port.data.get_shape()),
                                                 )).create_node()

                    # Reconnect input and weights to scale shift node
                    tensor_port.get_connection().set_destination(scsh_op.in_port(0))
                    value_port.get_connection().set_destination(scsh_op.in_port(1))
                    const_op.out_port(0).connect(scsh_op.in_port(2))
                else:
                    # Create fake weights for scale shift node
                    const_op = Const(graph, dict(name='{}/weights'.format(scsh_op.name),
                                                 value=np.ones(value_port.data.get_shape(), dtype=np.float32),
                                                 shape=np.array(value_port.data.get_shape()),
                                                 )).create_node()

                    # Reconnect input and biases to scale shift node
                    tensor_port.get_connection().set_destination(scsh_op.in_port(0))
                    const_op.out_port(0).connect(scsh_op.in_port(1))
                    value_port.get_connection().set_destination(scsh_op.in_port(2))

                node.out_port(0).get_connection().set_source(scsh_op.out_port(0))

                # Set bin attribute to ScaleShift input ports
                scsh_op.in_port(1).bin = 'weights'
                scsh_op.in_port(2).bin = 'biases'
    graph.strict_mode = True
