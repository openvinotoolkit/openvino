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
from collections import deque

import networkx as nx
import numpy as np

from mo.front.extractor import add_attrs_props
from mo.graph.graph import Node, unique_id
from mo.middle.passes.eliminate import graph_clean_up
from mo.utils.graph import pseudo_topological_sort
from mo.ops.lin_op import Mul, Add
from mo.ops.op import Op
from mo.graph.graph import dump_graph_for_graphviz
from mo.middle.passes.fusing.helpers import backward_bfs, forward_bfs, get_tensor_id, get_value_id


def _fuse_mul(graph: nx.MultiDiGraph, node: Node, fuse_nodes: list, backward: bool = True):
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
    const_id, tensor_id = get_value_id(node), get_tensor_id(node)

    if const_id is None or tensor_id is None:
        log.warning('Cannot do fuse_mul for node {} because this node has wrong inputs'.format(node.id))
        return False

    for fuse_node in fuse_nodes:
        if fuse_node.soft_get('can_be_fused') == False:
            log.warning('Node {} can\'t be used in fusing due to user specified attr can_be_fused = False'.format(fuse_node.id))
            return False

        if len(fuse_node.in_nodes()) < 2:
            log.warning('Node {} has no weights node'.format(fuse_node.id))
            return False

        if not fuse_node.has_valid('layout'):
            log.warning('Node {} has no layout attr'.format(fuse_node.id))
            return False

        weights_node = fuse_node.in_node(1)

        if not weights_node.has_valid('output_channel_dim') or not weights_node.has_valid('input_channel_dim'):
            log.warning(
                'Cannot do fuse_mul for node {} because there is no field ' +
                'output_channel_dim and/or input_channel_dim in weights.'
                .format(fuse_node.soft_get('name'))
            )
            return False

        inp_ch, out_ch = weights_node.input_channel_dim, weights_node.output_channel_dim
        if max(inp_ch, out_ch) >= len(weights_node.shape):
            log.warning('Node {} has wrong weights shape'.format(fuse_node.id))
            return False

    for fuse_node in fuse_nodes:
        weights_node = fuse_node.in_node(1)
        value = np.array(node.in_node(const_id).value)

        value = np.squeeze(value)

        # TODO : ch_dim should be equal to node.in_node(1).value.shape
        # We will multiply weights according output/input channel dimension
        ch_dim = weights_node.output_channel_dim if backward else weights_node.input_channel_dim
        shape = np.array([weights_node.shape[ch_dim]])

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
        wdims_number = weights_node.dims_number
        for x in range(wdims_number - ch_dim - 1):
            shape = np.append(shape, 1)

        mul_val = np.array(value)
        value = np.reshape(value, shape)

        # Weights multiplication
        weights_node.value = weights_node.value * value

        # If we fuse in backward direction we should multiply biases if they exists
        if backward and len(fuse_node.in_nodes()) == 3:
            conv_bias = fuse_node.in_node(2)
            conv_bias.value = conv_bias.value * np.squeeze(mul_val)
        log.debug('Fused: {} to {}'.format(node.name, fuse_node.name))
        is_fused = True

    if is_fused:
        # Delete Mul node
        out_node = node.out_node()
        op_data_node = node.in_node(tensor_id)
        op_const_node = node.in_node(const_id)
        op_node = op_data_node.in_node(0)
        graph.remove_edge(node.id, out_node.id)
        graph.remove_edge(op_node.id, op_data_node.id)
        graph.remove_edge(op_const_node.id, node.id)
        # Connect nodes after deleting
        graph.add_edge(op_node.id, out_node.id, out=0)
        for idx in reversed(range(len(op_data_node.out_nodes()))):
            out_data = op_data_node.out_nodes()[idx]
            edge_attrs = graph.get_edge_data(op_data_node.id, out_data.id)[0]
            if not out_data.id is node.id:
                graph.remove_edge(op_data_node.id, out_data.id)
                graph.add_edges_from([(out_node.id, out_data.id, edge_attrs)])

    return is_fused


def _fuse_add(graph: nx.MultiDiGraph, node: Node, fuse_nodes: list, backward: bool = True):
    """
    This function takes Add node and Convolution/FC nodes for further fusion and then deletes Add node
    In case if Convolution/FC Bias absence it will be created
    """
    is_fused = False
    const_id, tensor_id = get_value_id(node), get_tensor_id(node)

    if const_id is None or tensor_id is None:
        log.warning('Cannot do fuse_add for node {} because this node has wrong inputs'.format(node.id))
        return False

    # if len(node.in_node(const_id).shape) > 2 or any([x == 0 for x in node.in_node(const_id).shape]):
    #     log.warning('Cannot do fuse_add for node {} because this node has wrong shape'.format(node.id))
    #     return False

    for fuse_node in fuse_nodes:
        if fuse_node.soft_get('can_be_fused') == False:
            log.warning('Node {} can\'t be used in fusing due to user specified attr can_be_fused = False'.format(fuse_node.id))
            return False
        if not fuse_node.has_valid('layout'):
            log.warning('Node {} has no layout attr'.format(fuse_node.id))
            return False
        if len(fuse_node.in_nodes()) < 2:
            log.warning('Node {} has no weights node'.format(fuse_node.id))
            return False

    for fuse_node in fuse_nodes:
        value = np.array(node.in_node(const_id).value)

        # If forward, broadcast value
        if not backward:
            cnt = fuse_node.in_node(1).shape[-1] / node.in_node(const_id).shape[0]
            if fuse_node.layout == 'NCHW':
                tmp = []
                for val in value:
                    tmp = np.concatenate((tmp, np.repeat(val, cnt)))
                value = np.array(tmp)
            else:
                value = np.tile(value, int(cnt))

        value = np.squeeze(value)

        # Create BIAS data node if not exists
        if len(fuse_node.in_nodes()) <= 2:
            bias_data = unique_id(graph, "bias_data")
            data_type = fuse_node.in_node(1).data_type
            # Broadcast if scalar
            if value.size == 1:
                id = fuse_node.in_node(1).output_channel_dim if backward else fuse_node.in_node(1).input_channel_dim
                vshape = fuse_node.in_node(1).shape[id]
                value = np.full(vshape, value.item())

            if not backward:
                value = np.dot(fuse_node.in_node(1).value, value)

            shape = value.shape

            graph.add_node(bias_data, **add_attrs_props(
                dict(kind='data', precision="FP32", name=bias_data, value=value, shape=shape, data_type=data_type)))
            graph.add_edges_from([(bias_data, fuse_node.id, {'in': 2, 'bin': 'biases'})])
            fuse_node['bias_term'] = True
        else:
            if not backward:
                fuse_node.in_node(2).value += np.dot(fuse_node.in_node(1).value, value)
            else:
                fuse_node.in_node(2).value += value

        log.debug('Fused: {} to {}'.format(node.name, fuse_node.name))
        is_fused = True

    if is_fused:
        # Delete Add node
        out_node = node.out_node()
        op_data_node = node.in_node(tensor_id)
        op_const_node = node.in_node(const_id)
        op_node = op_data_node.in_node(0)
        graph.remove_edge(node.id, out_node.id)
        graph.remove_edge(op_node.id, op_data_node.id)
        graph.remove_edge(op_const_node.id, node.id)
        # Connect nodes after deleting
        graph.add_edge(op_node.id, out_node.id, out=0)
        for idx in reversed(range(len(op_data_node.out_nodes()))):
            out_data = op_data_node.out_nodes()[idx]
            edge_attrs = graph.get_edge_data(op_data_node.id, out_data.id)[0]
            if not out_data.id is node.id:
                graph.remove_edge(op_data_node.id, out_data.id)
                graph.add_edges_from([(out_node.id, out_data.id, edge_attrs)])

    return is_fused


def fuse_linear_ops(graph: nx.MultiDiGraph):
    """
    This function makes fusing of linear operations (Mul,Add) to Convolution/FC.
    """
    fuse_count = 0

    # Fusion in backward direction
    nodes = pseudo_topological_sort(graph)
    for idx in nodes:
        node = Node(graph, idx)
        is_fused = False

        # Fuse Mul to Convolution/FC
        if node.soft_get('op') == 'Mul' and get_value_id(node) is not None and node.soft_get('can_be_fused') == True:
            fuse_nodes = backward_bfs(node, [], ['Convolution', 'Deconvolution', 'FullyConnected'])
            is_fused = _fuse_mul(graph, node, fuse_nodes)

        # Fuse Add to Convolution/FC
        if node.soft_get('op') == 'Add' and get_value_id(node) is not None and node.soft_get('can_be_fused') == True:
            fuse_nodes = backward_bfs(node, [], ['Convolution', 'Deconvolution', 'FullyConnected'])
            is_fused = _fuse_add(graph, node, fuse_nodes)

        fuse_count += is_fused

    # Fusion in forward direction
    nodes = pseudo_topological_sort(graph, reverse=True)
    for idx in nodes:
        node = Node(graph, idx)
        is_fused = False

        # Fuse Mul to Convolution/FC
        if node.soft_get('op') == 'Mul' and get_value_id(node) is not None and node.soft_get('can_be_fused') == True:
            fuse_nodes = forward_bfs(node, [], ['Convolution', 'Deconvolution', 'FullyConnected'])
            is_fused = _fuse_mul(graph, node, fuse_nodes, False)

        # Fuse Add to Convolution/FC
        if node.soft_get('op') == 'Add' and get_value_id(node) is not None and node.soft_get('can_be_fused') == True:
            fuse_nodes = forward_bfs(node, [], ['FullyConnected'])
            is_fused = _fuse_add(graph, node, fuse_nodes, False)

        fuse_count += is_fused

    log.debug("Fused {} nodes".format(fuse_count))
