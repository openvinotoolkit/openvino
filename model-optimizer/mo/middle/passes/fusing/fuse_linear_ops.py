"""
 Copyright (C) 2018-2020 Intel Corporation

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

from mo.graph.graph import Node, Graph
from mo.middle.passes.fusing.helpers import backward_bfs, forward_bfs, get_value_in_port, \
    get_tensor_in_port
from mo.ops.const import Const


def _fuse_mul(graph: Graph, node: Node, fuse_nodes: list, backward: bool = True):
    """
    This function takes Mul node and array of convolution/fc nodes for further fusion
    Parameters
    ----------
    x : bool
        If backward is False, that means that Convolution/FC goes after Mul node
        else means that Mul goes after Convolutions/FC
        :param backward:
        :param fuse_nodes:
        :param node:
        :param graph:
    """
    is_fused = False
    const_port, tensor_port = get_value_in_port(node), get_tensor_in_port(node)

    if const_port is None or tensor_port is None:
        log.warning('Cannot do fuse_mul for node {} because this node has wrong inputs'.format(node.id))
        return False

    for fuse_node in fuse_nodes:
        if fuse_node.soft_get('can_be_fused') is False:
            log.warning('Node {} can\'t be used in fusing because attr can_be_fused = False'.format(fuse_node.name))
            return False

        if len(fuse_node.in_ports()) < 2:
            log.warning('Node {} has no weights node'.format(fuse_node.name))
            return False

        if not backward and not fuse_node.has_valid('layout'):
            log.warning('Node {} has no layout attr'.format(fuse_node.name))
            return False

        weights_port = fuse_node.in_port(1)
        if not weights_port.data.has_valid('output_channel_dim') or \
                not weights_port.data.has_valid('input_channel_dim'):
            log.warning(
                'Cannot do fuse_mul for node {} because there is no field ' +
                'output_channel_dim and/or input_channel_dim in weights.'
                .format(fuse_node.soft_get('name'))
            )
            return False

        inp_ch = weights_port.data.get_attr('input_channel_dim')
        out_ch = weights_port.data.get_attr('output_channel_dim')
        if max(inp_ch, out_ch) >= len(weights_port.data.get_shape()):
            log.warning('Node {} has wrong weights shape'.format(fuse_node.name))
            return False

    for fuse_node in fuse_nodes:
        weights_port = fuse_node.in_port(1)
        value = np.array(const_port.data.get_value())

        value = np.squeeze(value)

        # TODO : ch_dim should be equal to node.in_node(1).value.shape
        # We will multiply weights according output/input channel dimension
        ch_dim = weights_port.data.get_attr('output_channel_dim' if backward else 'input_channel_dim')
        shape = np.array([weights_port.data.get_shape()[ch_dim]])

        # Scalar broadcast
        if value.size == 1:
            value = np.full(shape, value.item())

        # Common broadcast for forward fusion
        if not backward:
            cnt = shape[-1] / value.shape[0]
            if fuse_node.layout == 'NCHW':
                tmp = []
                for val in value:
                    tmp = np.concatenate((tmp, np.repeat(val, cnt)))
                value = np.array(tmp)
            else:
                value = np.tile(value, int(cnt))

        # Expand dims for multiplication (ex. [38] to [38, 1, 1])
        wdims_number = weights_port.data.get_attr('dims_number')
        for x in range(wdims_number - ch_dim - 1):
            shape = np.append(shape, 1)

        mul_val = np.array(value)
        value = np.reshape(value, shape)

        # Weights multiplication
        mul_const = Const(graph, {'value': value}).create_node()
        w_mul = node.copy_node({'in_ports_count': len(node.in_ports()), 'out_ports_count': len(node.out_ports()),
                                'can_be_fused': False})
        w_mul.in_port(const_port.idx).connect(mul_const.out_port(0))
        w_const = weights_port.get_source()
        weights_port.get_connection().set_source(w_mul.out_port(0))
        w_const.connect(w_mul.in_port(tensor_port.idx))

        # If we fuse in backward direction we should multiply biases if they exists
        if backward and len(fuse_node.in_ports()) == 3 and not fuse_node.in_port(2).disconnected() and \
                not fuse_node.has_and_set('shape_input'):
            conv_bias = fuse_node.in_port(2)
            conv_bias.data.set_value(conv_bias.data.get_value() * np.squeeze(mul_val))

        mul_const.infer(mul_const)
        w_mul.infer(w_mul)

        log.debug('Fused: {} to {}'.format(node.name, fuse_node.name))
        is_fused = True

    if is_fused:
        # Delete Mul node
        producer_port = tensor_port.get_source()
        tensor_port.disconnect()
        const_port.disconnect()
        node.out_port(0).get_connection().set_source(producer_port)

    return is_fused


def fuse_linear_ops(graph: Graph):
    """
    This function makes fusing of linear operations (Mul,Add) to Convolution/FC.
    """
    fuse_count = 0

    # Fusion in backward direction
    nodes = graph.pseudo_topological_sort()
    for node in nodes:
        is_fused = False

        # Fuse Mul to Convolution/FC
        if node.soft_get('op') == 'Mul' and get_value_in_port(node) is not None and node.has_and_set('can_be_fused'):
            fuse_nodes = backward_bfs(node, [], ['Convolution', 'Deconvolution', 'MatMul'])
            is_fused = _fuse_mul(graph, node, fuse_nodes)

        fuse_count += is_fused

    # Fusion in forward direction
    nodes = graph.pseudo_topological_sort(reverse=True)
    for node in nodes:
        is_fused = False

        # Fuse Mul to Convolution/FC
        if node.soft_get('op') == 'Mul' and get_value_in_port(node) is not None and node.has_and_set('can_be_fused'):
            fuse_nodes = forward_bfs(node, [], ['Convolution', 'Deconvolution', 'MatMul'])
            is_fused = _fuse_mul(graph, node, fuse_nodes, False)

        fuse_count += is_fused

    log.debug("Fused {} nodes".format(fuse_count))
