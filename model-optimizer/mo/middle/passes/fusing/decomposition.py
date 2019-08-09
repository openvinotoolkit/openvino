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

from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.graph.port import Port
from mo.middle.pattern_match import apply_pattern
from mo.ops.const import Const
from extensions.ops.elementwise import Mul, Add
from mo.ops.op import Op
from mo.ops.reshape import Reshape


def expand_node_shape(port: Port, broadcast_dims_cnt):
    value = np.array(port.data.get_value())
    for idx in range(broadcast_dims_cnt):
        value = np.expand_dims(value, axis=-1)
    port.data.set_value(value)


def convert_batch_norm(graph: Graph):
    """
    This function finds FusedBatchNorm layer (or BatchNorm for MXNet) and replaces with Mul->Add->Mul->Add sequence.
    """
    nodes = graph.get_op_nodes()
    for node in nodes:
        if node.has_valid('op') and (node.op in ['FusedBatchNorm', 'BatchNorm', 'BatchNormalization']):

            if any([node.in_port(i).data.get_value() is None for i in range(1, len(node.in_ports()))]):
                log.warning('Cannot translate FusedBatchNorm {} node with non-constant weights'.format(
                    node.name if node.has_valid('name') else '<UNKNOWN>'))
                continue

            const = node.in_port(1).get_source()
            node.in_port(1).disconnect()

            beta = node.in_port(2).get_source()
            node.in_port(2).disconnect()

            mean = node.in_port(3).get_source()
            node.in_port(3).disconnect()

            variance = node.in_port(4).get_source()
            node.in_port(4).disconnect()

            eps = node.eps

            if node.has_valid('fix_gamma') and node.fix_gamma:
                const.data.get_value().fill(1.)

            can_be_fused = False if not node.soft_get('can_be_fused') else True

            scale = 1. / np.sqrt(variance.data.get_value() + eps)
            shift = (mean.data.get_value() * (-1)) * scale

            # Expand dims for current layout
            broadcast_dims_cnt = len(node.in_port(0).data.get_shape()) - 2 if graph.graph['layout'] == 'NCHW' else 0

            # Update values and shapes with new shape
            expand_node_shape(const, broadcast_dims_cnt)
            expand_node_shape(beta, broadcast_dims_cnt)

            for idx in range(broadcast_dims_cnt):
                scale = np.expand_dims(scale, axis=-1)
                shift = np.expand_dims(shift, axis=-1)

            _fused_batch_norm_decomposition(graph, node.in_port(0), node.out_port(0), const, beta, scale, shift, can_be_fused)


def _fused_batch_norm_decomposition(graph: Graph, tinput: Port, toutput: Port, gamma: Port, beta: Port,
                                    mean: np.ndarray, variance: np.ndarray, can_be_fused=True):
    """
    This is common function for TF, Caffe and MXNet
    It creates Mul->Add->Mul->Add sub graph
    """
    shape = tinput.data.get_shape()

    # Create first Mul & Add operations
    mul1_node = Mul(graph, dict(name="Mul1_", can_be_fused=can_be_fused)).create_node()
    add1_node = Add(graph, dict(name="Add1_", can_be_fused=can_be_fused)).create_node()

    const_mul1_node = Const(graph, dict(name="data_mul_", value=np.array(mean))).create_node()
    const_add1_node = Const(graph, dict(name="data_add_", value=np.array(variance))).create_node()

    # Broadcast const from scalar
    # We can broadcast only when const.value is scalar
    if gamma.data.get_shape()[0] != gamma.data.get_value().shape[0]:
        value = gamma.data.get_value()
        value.resize(gamma.data.get_shape()).fill(value[0])
        gamma.data.set_value(value)

    # Create second Mul & Add
    mul2_node = Mul(graph, dict(name="Mul2_", can_be_fused=can_be_fused)).create_node()
    add2_node = Add(graph, dict(name="Add2_", can_be_fused=can_be_fused)).create_node()

    # Connect edges Mul1->Add1->Mul2->Add2
    tinput.get_connection().set_destination(mul1_node.in_port(0))
    mul1_node.in_port(1).get_connection().set_source(const_mul1_node.out_port(0))

    add1_node.in_port(0).get_connection().set_source(mul1_node.out_port(0))
    add1_node.in_port(1).get_connection().set_source(const_add1_node.out_port(0))

    mul2_node.in_port(0).get_connection().set_source(add1_node.out_port(0))
    gamma.get_connection().set_destination(mul2_node.in_port(1))

    add2_node.in_port(0).get_connection().set_source(mul2_node.out_port(0))
    beta.get_connection().set_destination(add2_node.in_port(1))

    toutput.get_connection().set_source(add2_node.out_port(0))


def convert_scale_shift_to_mul_add(graph: Graph):
    nodes = graph.get_op_nodes(op='ScaleShift')
    for node in nodes:
        if node.soft_get('can_be_fused') is False:
            continue

        ports_count = len(node.in_ports())

        input_port = node.in_port(0)
        scale_port = node.in_port(1) if ports_count > 1 and not node.in_port(1).disconnected() else None
        shift_port = node.in_port(2) if ports_count > 2 and not node.in_port(2).disconnected() else None
        output_port = node.out_port(0)

        has_biases = True
        has_weights = True

        # We don't need zero biases
        if shift_port is None or (shift_port.data.get_value() is not None and all([x == 0 for x in shift_port.data.get_value()])):
            has_biases = False

        # We don't need weights with ones
        if scale_port is None or (scale_port.data.get_value() is not None and all([x == 1 for x in scale_port.data.get_value()])):
            has_weights = False

        mul_op = Mul(graph, dict(name=node.name + "/Mul_"))
        add_op = Add(graph, dict(name=node.name + "/Add_"))

        # Expand dims for current layout
        broadcast_dims_cnt = len(input_port.data.get_shape()) - 2 if graph.graph['layout'] == 'NCHW' else 0

        # In case if we have constant weights/biases we have to broadcast them according to graph layout
        # otherwise we insert Reshape with broadcast dim attribute.
        def broadcast_value(port):
            value = np.array(port.data.get_value())
            for idx in range(broadcast_dims_cnt):
                value = np.expand_dims(value, axis=-1)
            port.data.set_value(value)

        def broadcast_with_reshape(port):
            input_shape = input_port.data.get_shape()
            reshape_dims = np.zeros(len(input_shape), dtype=np.int64)
            for i in range(0, node.axis):
                reshape_dims[i] = 1
            data_shape = port.data.get_shape()
            for i in range(node.axis, node.axis + len(data_shape)):
                reshape_dims[i] = data_shape[i - node.axis]
            for i in range(node.axis + len(data_shape), len(input_shape)):
                reshape_dims[i] = 1
            reshape = create_op_node_with_second_input(graph, Reshape, reshape_dims,
                                                       dict(name=port.node.name + "/Broadcast_"))
            port.get_connection().set_destination(reshape.in_port(0))
            reshape.out_port(0).connect(port)

        if has_weights and scale_port.data.get_value() is not None:
            broadcast_value(scale_port)
        elif has_weights:
            broadcast_with_reshape(scale_port)

        if has_biases and shift_port.data.get_value() is not None:
            broadcast_value(shift_port)
        elif has_biases:
            broadcast_with_reshape(shift_port)

        if has_biases and has_weights:
            # Connect input->mul->out->add->out
            add_node = add_op.create_node()
            mul_node = mul_op.create_node()

            # Connect Mul operation with inputs
            input_port.get_connection().set_destination(mul_node.in_port(0))
            scale_port.get_connection().set_destination(mul_node.in_port(1))

            # Connect Add operation with inputs
            mul_node.out_port(0).connect(add_node.in_port(0))
            shift_port.get_connection().set_destination(add_node.in_port(1))

            output_port.get_connection().set_source(add_node.out_port(0))
        elif has_weights:
            # Connect input->mul->out
            mul_node = mul_op.create_node()

            # Connect Mul operation with inputs
            input_port.get_connection().set_destination(mul_node.in_port(0))
            scale_port.get_connection().set_destination(mul_node.in_port(1))

            output_port.get_connection().set_source(mul_node.out_port(0))
        elif has_biases:
            # Connect input->add->out
            add_node = add_op.create_node()

            # Connect Add operation with inputs
            input_port.get_connection().set_destination(add_node.in_port(0))
            shift_port.get_connection().set_destination(add_node.in_port(1))

            output_port.get_connection().set_source(add_node.out_port(0))
        else:
            # Connect input->out
            producer_port = input_port.get_source()
            input_port.disconnect()
            output_port.get_connection().set_source(producer_port)


def _bn_to_mul_add_action(graph: Graph, match: dict):
    # Data nodes
    tinput = match['input']
    toutput = match['output']
    mean = match['mean']
    variance = match['variance']

    # Op node
    bn_node = match['batch_norm']

    # Disconnect data nodes from
    graph.remove_edge(tinput.node, bn_node.node)
    graph.remove_edge(mean.node, bn_node.node)
    graph.remove_edge(variance.node, bn_node.node)

    graph.remove_edge(bn_node.node, toutput.node)

    scale = 1. / np.sqrt(variance.value + bn_node.epsilon)
    shift = (mean.value * (-1)) * scale

    mean.value = np.array(scale)
    variance.value = np.array(shift)

    # Expand dims for current layout
    broadcast_dims_cnt = len(tinput.shape) - 2 if graph.graph['layout'] == 'NCHW' else 0
    # Update values and shapes with new shape
    Op.expand_node_shape(mean, broadcast_dims_cnt)
    Op.expand_node_shape(variance, broadcast_dims_cnt)

    can_be_fused = False if not bn_node.soft_get('can_be_fused') else True

    mul_node = Mul(graph, dict(name="Mul_", can_be_fused=can_be_fused))
    add_node = Add(graph, dict(name="Add_", can_be_fused=can_be_fused))

    # Connect input->mul->add
    add_node.create_node_with_data(inputs=[mul_node.create_node_with_data(inputs=[tinput, mean]), variance],
                                   data_nodes=toutput)


def convert_bn_to_mul_add(graph: Graph):
    apply_pattern(
        graph,
        nodes=[
            ('input', dict(kind='data')),
            ('mean', dict(kind='data')),
            ('variance', dict(kind='data')),
            ('output', dict(kind='data')),
            ('batch_norm', dict(kind='op', op='BatchNormalization')),
        ],
        edges=[
            ('input', 'batch_norm', {'in': 0}),
            ('mean', 'batch_norm', {'in': 1}),
            ('variance', 'batch_norm', {'in': 2}),
            ('batch_norm', 'output'),
        ],
        action=_bn_to_mul_add_action
    )
