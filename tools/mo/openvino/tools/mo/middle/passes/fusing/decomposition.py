# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.graph.port import Port
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.reshape import Reshape


def expand_node_shape(port: Port, broadcast_dims_cnt):
    value = mo_array(port.data.get_value())
    for idx in range(broadcast_dims_cnt):
        value = np.expand_dims(value, axis=-1)
    port.data.set_value(value)


def convert_batch_norm(graph: Graph):
    """
    This function finds FusedBatchNorm layer (or BatchNorm for MXNet) and replaces with Mul->Add->Mul->Add sequence.
    """
    nodes = graph.get_op_nodes()
    for node in nodes:
        if node.has_valid('op') and (node.op in ['FusedBatchNorm', 'FusedBatchNormV2', 'FusedBatchNormV3',
                                                 'BatchNorm', 'BatchNormalization', 'batchNormInference']):

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
            layout = node.soft_get('data_format', graph.graph['layout'])
            broadcast_dims_cnt = len(node.in_port(0).data.get_shape()) - 2 if layout in ['NCHW', "NCDHW"] else 0

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
    This is common function for TF and Caffe
    It creates Mul->Add->Mul->Add sub graph
    """
    batch_norm_name = tinput.get_connection().get_destination().node.name

    # Create first Mul & Add operations
    mul1_node = Mul(graph, dict(name=batch_norm_name + "/mean", can_be_fused=can_be_fused)).create_node()
    add1_node = Add(graph, dict(name=batch_norm_name + "/variance", can_be_fused=can_be_fused)).create_node()

    const_mul1_node = Const(graph, dict(name="data_mul_", value=mo_array(mean))).create_node()
    const_add1_node = Const(graph, dict(name="data_add_", value=mo_array(variance))).create_node()

    # Broadcast const from scalar
    # We can broadcast only when const.value is scalar
    if gamma.data.get_shape()[0] != gamma.data.get_value().shape[0]:
        value = gamma.data.get_value()
        value.resize(gamma.data.get_shape()).fill(value[0])
        gamma.data.set_value(value)

    # Create second Mul & Add
    mul2_node = Mul(graph, dict(name=batch_norm_name + "/gamma", can_be_fused=can_be_fused)).create_node()
    add2_node = Add(graph, dict(name=batch_norm_name + "/beta", can_be_fused=can_be_fused)).create_node()

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
            value = mo_array(port.data.get_value())
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
