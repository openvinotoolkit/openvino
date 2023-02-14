# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.tools.mo.graph.graph import Node


def get_node_inputs(node: Node):
    """
    Return list of node input nodes
    Indexes of input nodes in list matches number of input port for this inputs
    :param node: node from NetworkX to get inputs
    :return: list of node inputs
    """
    return [port.node if port is not None else None for port in get_node_input_ports(node)]


def get_node_input_ports(node: Node):
    """
    Return list of node input nodes with their ports
    Indexes of input nodes in list matches number of input port for this inputs
    :param node: node from NetworkX to get inputs
    :return: list of node inputs
    """
    sources_ports = [parent.get_source() for parent in node.in_ports().values()]
    return [port for port in sources_ports if port is not None]


def get_node_input(node: Node, in_port: int):
    """
    Returns input node of node on in_port
    :param node: node from NetworkX to get input node
    :param in_port: number of input port
    :return: input node of 'node' on in_port input port
    """
    out_port = node.in_port(in_port).get_source()
    return out_port.node if out_port else None


def get_all_node_outputs(node: Node):
    """
    Returns all consumers of node outputs
    WARNING: Indexes of consumer nodes in list DOESN'T MATCH indexes
    of output ports for this consumers (because one port can have more than one consumer).
    :param node: NetworkX node to get outputs
    :return list of all node consumer nodes
    """
    return [port.node for port in get_node_output_ports(node)]


def get_node_output_ports(node: Node):
    """
    Returns all consumers of node outputs
    WARNING: Indexes of consumer nodes in list DOESN'T MATCH indexes
    of output ports for this consumers (because one port can have more than one consumer).
    :param node: NetworkX node to get outputs
    :return list of all node consumer nodes
    """
    consumers = []
    for port in node.out_ports().values():
        for dst_port in port.get_destinations():
            consumers.append(dst_port)
    return consumers


def get_node_output(node: Node, out_port: int):
    """
    Return all consumers of node from output port out_port
    :param node: node from NetworkX to get output on port
    :param out_port: number of output port to get consumers
    :return: List of consumers of out_port of node
    """
    consumers = []
    for dst_node in node.out_port(out_port).get_destinations():
        consumers.append(dst_node.node)
    return consumers


def set_node_value(node: Node, value: np.ndarray):
    """
    Set new value to Const node and recompute all necessary
     shapes in this node and data nodes
     :param node: node to set value
     :param value: new node value
      """
    if node.type != 'Const':
        raise Exception('Can\'t set value for non-constant node {}'.format(node.name))
    data_type = np.float32
    if node.out_port(0).is_data_type_defined():
        data_type = node.out_port(0).get_data_type()
    node.out_port(0).data.set_value(np.array(value).astype(data_type))


def get_node_value(node: Node):
    """
    Get value from Const node
    :param node: node to get value
    :return the value of the Const node
    """
    if node.type != 'Const':
        raise Exception('Can\'t get value for non-constant node {}'.format(node.name))
    return node.value


def get_weights_for_node(node: Node):
    """
    Return node with weights for node
    :param node: NetworkX node to get weights
    :return: node with weights and None if no weights for node
    """

    if 1 not in node.in_ports():
        raise Exception('Can\'t get weights for {} node. No 1 port in node'.format(node.name))

    if node.in_port(1).get_source() is not None:
        return node.in_port(1).get_source().node
    return None


def set_weights_for_node(node: Node, value: np.ndarray):
    """
    Set new value of weights for node
    :param node:  NetworkX node to set new weights value
    :param value: new weights value
    :return: None
    """
    weights = get_weights_for_node(node)
    if weights is None:
        raise Exception('Can\'t set weights for node {} because node does not have weights'.format(node.name))
    set_node_value(weights, value)


def get_bias_for_node(node: Node):
    """
    Return node with bias for node
    :param node: NetworkX node to get bias
    :return: node with bias, None if there is no bias for node
    """
    node_outputs = get_node_output(node, 0)
    if len(node_outputs) == 1:
        potential_bias = node_outputs[0]
        if potential_bias.type == 'Add' and len(get_node_inputs(potential_bias)) > 1:
            for potential_bias_const in get_node_inputs(potential_bias):
                if potential_bias_const.type == 'Const':
                    return potential_bias_const
    return None


def set_bias_for_node(node: Node, value: np.ndarray):
    """
    Set new value of bias for node
    :param node:  NetworkX node to set new bias value
    :param value: new bias value
    :return: None
    """
    bias = get_bias_for_node(node)
    if bias is None:
        raise Exception('Can\'t set bias for node {} because node does not have a bias'.format(node.name))
    set_node_value(bias, value)


def get_input_shape(node: Node, in_port: int):
    """
    Return shape of in_port input of node
    :param node: NetworkX node to get input shape
    :param in_port: input port number
    :return:
    """
    if in_port not in node.in_ports():
        raise Exception('Can\'t get shape for {} port of {} node. No such port in node'.format(in_port, node.name))
    in_port = node.in_port(in_port)
    return in_port.data.get_shape()


def get_output_shape(node: Node, out_port: int):
    """
    Return shape of out_port input of node
    :param node: NetworkX node to get input shape
    :param in_port: input port number
    :return:
    """
    if out_port not in node.out_ports():
        raise Exception('Can\'t get shape for {} port of {} node. No such port in node'.format(out_port, node.name))
    out_port = node.out_port(out_port)
    return out_port.data.get_shape()


def get_quantized_input_key(quantized_node):
    """
    Returns key for quantized node input.
    If input node of quantized node have one output port -> key is name of fq_input node.
    Otherwise, key is tuple (fq_input name, output port number)
    """
    if quantized_node.type == 'Add':
        for quantized_node_input in get_node_inputs(quantized_node):
            if quantized_node_input.type != 'Const':
                quantized_input = quantized_node_input
    else:
        quantized_input = get_node_input(quantized_node, 0)
    key = quantized_input.fullname
    if len(quantized_input.out_ports()) > 1:
        port_number = quantized_node.in_port(0).get_source().out
        key = (quantized_input.fullname, port_number)
    return key


def node_with_quantized_weights(node):
    """
    Check that node have constant and quantized weights (input on port 1).
    :param node: operation node
    :return: True if node has quantized weights and False instead
    """
    weights_input = get_node_input(node, 1)
    if weights_input.type == 'FakeQuantize' and get_node_input(weights_input, 0).type == 'Const':
        return True

    return False


def get_input_data_value(node: Node, port: int):
    """
    Return value of data node for needed node at port
    :param node: needed node
    :param port: input port id
    :return: input data node value
    """
    return node.in_port(port).data.get_value()


def get_first_convolutions(parameter_nodes):
    first_convolutions = []
    while parameter_nodes:
        first_convolutions += [node for node in parameter_nodes
                               if node.type == 'Convolution']
        parameter_nodes = [get_all_node_outputs(node) for node in parameter_nodes
                           if node.type != 'Convolution']
        parameter_nodes = {node for node_list in parameter_nodes for node in node_list}
    return first_convolutions


def check_const_input(node):
    w_out = get_weights_for_node(node)
    if w_out.type == 'FakeQuantize':
        w_out = get_node_input(w_out, 0)
    if w_out.type != 'Const':
        return False
    return True


def check_input_data_is_const(node, port_id=1):
    return get_node_input(node, port_id).type == 'Const' or get_input_data_value(node, port_id) is not None


def get_lstm_ends(read_value, assigns, ignore_nodes):
    assigns_by_id = {n.variable_id: n for n in assigns}
    assign = assigns_by_id[read_value.variable_id]
    assign_input = get_node_input(assign, 0)
    lstm_outputs = [n for n in get_all_node_outputs(assign_input)
                    if n.name not in ignore_nodes]
    return lstm_outputs


def create_node_name(input_node, mode=tuple):
    """
    Returns key for node input.
    If input node has one output port -> key is name of input node.
    Otherwise, key is tuple (input name, output port number)
    """
    key = input_node.fullname
    if len(input_node.out_ports()) > 1:
        port_number = input_node.in_port(0).get_source().out
        key = (input_node.fullname, port_number) if mode == tuple else f"{input_node.fullname}.{port_number}"
    return key


def get_node_data_type(node, port_id=0):
    if node.type != 'Const' and port_id in node.in_ports() \
            and node.in_port(port_id).get_source() is not None \
            and node.in_port(port_id).get_source().is_data_type_defined():
        return node.in_port(port_id).get_source().get_data_type()
    return None


def reset_node_fullname(old_fullname, node_name):
    return '|'.join(old_fullname.split('|')[:-1] + [node_name])

def convert_to_outputs_name(node_name):
    return node_name if isinstance(node_name, tuple) else (node_name, 0)
