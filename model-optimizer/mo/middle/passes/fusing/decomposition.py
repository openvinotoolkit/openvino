"""
 Copyright (C) 2018-2021 Intel Corporation

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

import numpy as np

from extensions.ops.elementwise import Mul, Add, Pow
from extensions.ops.range import Range
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph, Node
from mo.graph.port import Port
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.unsqueeze import Unsqueeze


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
        if node.has_valid('op') and (node.op in ['BatchNorm', 'BatchNormalization', 'BatchNormInference',
                                                 'BatchNormInferenceMO']):

            # if any([node.in_port(i).data.get_value() is None for i in range(1, len(node.in_ports()))]):
            #     log.warning('Cannot translate FusedBatchNorm {} node with non-constant weights'.format(
            #         node.name if node.has_valid('name') else '<UNKNOWN>'))
            #     continue

            const = node.in_node(1)
            const_data = node.in_port(1).get_source()
            node.in_port(1).disconnect()

            beta = node.in_node(2)
            beta_data = node.in_port(2).get_source()
            node.in_port(2).disconnect()
            mean = node.in_node(3)
            mean_data = node.in_port(3).get_source()
            node.in_port(3).disconnect()

            variance = node.in_node(4)
            variance_data = node.in_port(4).get_source()
            node.in_port(4).disconnect()

            eps = node.eps
            var_is_const = variance_data.data.get_value() is not None
            mean_is_const = mean_data.data.get_value() is not None

            if node.has_valid('fix_gamma') and node.fix_gamma:
                const_data.data.get_value().fill(1.)

            # Broadcast const from scalar
            # We can broadcast only when const.value is scalar
            if const_data.data.get_shape()[0] != const_data.data.get_value().shape[0]:
                value = const_data.data.get_value()
                value.resize(const_data.data.get_shape()).fill(value[0])
                const_data.data.set_value(value)

            can_be_fused = False if not node.soft_get('can_be_fused') else True
            broadcast_dims_cnt = len(node.in_port(0).data.get_shape()) - 2 if graph.graph['layout'] == 'NCHW' else 0

            scale = None
            shift = None
            if var_is_const and mean_is_const:
                scale_val = np.float32(1. / np.sqrt(variance_data.data.get_value() + eps))
                shift_val = np.float32((mean_data.data.get_value() * (-1.)) * scale_val)

                for idx in range(broadcast_dims_cnt):
                    scale_val = np.expand_dims(scale_val, axis=-1)
                    shift_val = np.expand_dims(shift_val, axis=-1)

                scale = Const(graph, dict(name=node.name + "/scale_const", value=scale_val, can_be_fused=can_be_fused))\
                    .create_node_with_data()

                shift = Const(graph, dict(name=node.name + "/shift_const", value=shift_val, can_be_fused=can_be_fused))\
                    .create_node_with_data()

            if scale is None:
                neg_half_const = Const(graph,
                                       dict(name=node.name + "/neg_half_const", value=np.float32(-0.5),
                                            can_be_fused=can_be_fused)).create_node_with_data()
                eps_const = Const(graph, dict(name=node.name + "/eps_const", value=np.float32(eps),
                                              can_be_fused=can_be_fused)).create_node_with_data()

                var_add_eps = Add(graph, dict(name=node.name + "/var_add_eps",
                                              can_be_fused=can_be_fused)).create_node_with_data([variance, eps_const])
                scale = Pow(graph, dict(name=node.name + "/scale", can_be_fused=can_be_fused)).create_node_with_data([
                    var_add_eps, neg_half_const])

            if shift is None:
                neg_const = Const(graph, dict(name=node.name + "/neg_const", value=np.float32(-1.),
                                              can_be_fused=can_be_fused)).create_node_with_data()
                neg_mul_mean = Mul(graph, dict(name=node.name + "/neg_mul_mean", can_be_fused=can_be_fused)). \
                    create_node_with_data([mean, neg_const])
                shift = Mul(graph, dict(name=node.name + "/shift", can_be_fused=can_be_fused)).create_node_with_data(
                    [neg_mul_mean, scale])

            # Expand dims for current layout

            # Update values and shapes with new shape
            if broadcast_dims_cnt != 0:
                expand_node_shape(const_data, broadcast_dims_cnt)
                expand_node_shape(beta_data, broadcast_dims_cnt)
                if not (var_is_const and mean_is_const):
                    expand_dims_nodes_preff = node.name + '/expand_dims'
                    broad_cast_dims_const = Const(graph, dict(name=expand_dims_nodes_preff + "/broad_cast_dims_const",
                                                              value=np.int32(broadcast_dims_cnt))).create_node()
                    one_const = Const(graph, dict(name=expand_dims_nodes_preff + "/one_const",
                                                  value=np.int32(broadcast_dims_cnt))).create_node()
                    one_const.add_output_port(1, True)

                    expand_dims_range = Range(graph, dict(name=expand_dims_nodes_preff + '/range')).create_node()
                    expand_dims_range.in_port(0).get_connection().set_source(one_const.out_port(0))
                    expand_dims_range.in_port(1).get_connection().set_source(broad_cast_dims_const.out_port(0))
                    expand_dims_range.in_port(2).get_connection().set_source(one_const.out_port(1))
                    expand_dims_range.add_output_port(1, True)

                    new_scale = Unsqueeze(graph, dict(name=expand_dims_nodes_preff + '/scale_unsqeeze')).create_node()
                    new_scale.in_port(0).get_connection().set_source(scale.out_port(0))
                    new_scale.in_port(1).get_connection().set_source(expand_dims_range.out_port(0))

                    new_shift = Unsqueeze(graph, dict(name=expand_dims_nodes_preff + '/shift_unsqeeze')).create_node()
                    new_shift.in_port(0).get_connection().set_source(shift.out_port(0))
                    new_shift.in_port(1).get_connection().set_source(expand_dims_range.out_port(1))
                    scale = new_scale
                    shift = new_shift

            _fused_batch_norm_decomposition(graph, node.in_node(0), node.out_node(0), const, beta, scale, shift,
                                            node.name,
                                            can_be_fused)

            graph.remove_node(node.id)


def _fused_batch_norm_decomposition(graph: Graph, tinput: Node, toutput: Node, gamma: Node, beta: Node,
                                    mean: Node, variance: Node, node_name: str, can_be_fused=True):
    """
    This is common function for TF, Caffe and MXNet
    It creates Mul->Add->Mul->Add sub graph
    """
    batch_norm_name = node_name

    # Create first Mul & Add operations
    mul1_node = Mul(graph, dict(name=batch_norm_name + "/mean", can_be_fused=can_be_fused)). \
        create_node_with_data([tinput, mean])
    add1_node = Add(graph, dict(name=batch_norm_name + "/variance", can_be_fused=can_be_fused)). \
        create_node_with_data([mul1_node, variance])

    # Create second Mul & Add
    mul2_node = Mul(graph, dict(name=batch_norm_name + "/gamma", can_be_fused=can_be_fused)). \
        create_node_with_data([add1_node, gamma])
    Add(graph, dict(name=batch_norm_name + "/beta", can_be_fused=can_be_fused)). \
        create_node_with_data([mul2_node, beta], data_nodes=toutput)


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
        if shift_port is None or (
                shift_port.data.get_value() is not None and all([x == 0 for x in shift_port.data.get_value()])):
            has_biases = False

        # We don't need weights with ones
        if scale_port is None or (
                scale_port.data.get_value() is not None and all([x == 1 for x in scale_port.data.get_value()])):
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
