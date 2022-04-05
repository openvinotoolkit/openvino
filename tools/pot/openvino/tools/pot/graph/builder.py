# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

from openvino.tools.mo.graph.graph import Graph

from openvino.tools.pot.graph.node_utils import get_node_data_type, get_node_input, get_node_inputs
from .editor import create_node, connect_nodes_by_name, get_node_by_name


def build_graph(graph_attrs, meta_data, nodes, edges):
    """ Build the Graph with specific nodes and edges.
     :param graph_attrs: dictionary with graph attributes
     :param nodes: list of nodes where each node is tuple (node_name, type, attrs)
                  nodes=[
                      ('input', 'Parameter',  {}),
                      ('weights', 'Const', {}),
                      ('conv', 'Convolution', {}),
                      ('output', 'Result', {})
                  ]
     :param edges: list of edges where each edge is tuple (node_out, node_in, attrs)
                  edges=[
                      ('input', 'conv', {'out': 0, 'in': 0}),
                      ('weights', 'conv', {'out': 0, 'in': 1}),
                      ('conv', 'output', {'out': 0, 'in': 0})
                  ]
     :return: generated graph.
    """
    graph = Graph()
    graph.graph = graph_attrs
    graph.meta_data = meta_data

    for node in nodes:
        create_node(graph, node[0], node[1], node[2])

    for edge in edges:
        out_port = edge[2].get('out', 0)
        in_port = edge[2].get('in', 0)
        connect_nodes_by_name(graph, edge[0], out_port, edge[1], in_port)

    graph.clean_up()

    return graph


def make_copy_fake_quantize(nodes, edges, fq):
    weights, input_low, input_height, output_low, output_height = get_node_inputs(fq)

    fq_attrs = deepcopy(fq.attrs())
    if fq.has_valid('levels'):
        fq_attrs['levels'] = int(fq_attrs['levels'])

    nodes.extend([
        (fq.name, fq.type, fq_attrs),
        (input_low.name, input_low.type,
         {'value': input_low.value}),
        (input_height.name, input_height.type,
         {'value': input_height.value}),
        (output_low.name, output_low.type,
         {'value': output_low.value}),
        (output_height.name, output_height.type,
         {'value': output_height.value}),
        (weights.name, weights.type, {'value': weights.value.copy()})])

    edges.extend([
        (weights.name, fq.name, {'out': 0, 'in': 0}),
        (input_low.name, fq.name, {'out': 0, 'in': 1}),
        (input_height.name, fq.name, {'out': 0, 'in': 2}),
        (output_low.name, fq.name, {'out': 0, 'in': 3}),
        (output_height.name, fq.name, {'out': 0, 'in': 4})
    ])
    return fq.name


def make_copy_graph_attrs(model, input_name, input_shape):
    graph_attrs = deepcopy(model.graph)
    meta_data = deepcopy(model.meta_data)

    # if 'user_shapes' in graph_attrs and graph_attrs['user_shapes'] is not None:
    #     graph_attrs['user_shapes'][graph_attrs['inputs'][0]][0]['shape'] = input_shape
    graph_attrs['inputs'] = [input_name]
    graph_attrs['cmd_params'].mean_values = None
    graph_attrs['cmd_params'].placeholder_shapes = input_shape
    graph_attrs['cmd_params'].input_shape = '[{}]'.format(','.join([str(v) for v in input_shape]))
    graph_attrs['cmd_params'].scale_values = None
    graph_attrs['cmd_params'].mean_scale_values = None

    meta_data['mean_values'] = None
    meta_data['placeholder_shapes'] = input_shape
    meta_data['input_shape'] = '[{}]'.format(','.join([str(v) for v in input_shape]))
    meta_data['scale_values'] = None
    meta_data['mean_scale_values'] = None
    return graph_attrs, meta_data


def build_graph_for_node(model, input_name, input_shape, node, remove_bias=False, remove_fake_quantize=False):
    """ Build the Graph (input - node - output). The Convolution, FullyConnected node types are supported.
     :param model: source model
     :param input_name: name of the input node in the generated graph
     :param input_shape: shape of the input node in the generated graph
     :param node: node for which graph (input - node - output) will be generated
     :param remove_bias: remove bias in the generated graph
     :param remove_fake_quantize: remove fake quantize nodes in the generated graph
     :return: generated graph.
    """
    input_data_type = get_node_data_type(node, 0)
    nodes, edges = [], []
    nodes.append((input_name, 'Parameter', {'name': input_name, 'shape': input_shape,
                                            'type': 'Parameter', 'data_type': input_data_type}))

    node_attrs = deepcopy(node.attrs())
    if node.has_valid('output') and node.has_valid('get_output_feature_dim'):
        node_attrs['get_output_feature_dim'] = None

    nodes.append((node.name, node.type, node_attrs))
    edges.append((input_name, node.name, {'out': 0, 'in': 0}))

    parent_nodes = get_node_inputs(node)
    if parent_nodes[1].type == 'FakeQuantize' and not remove_fake_quantize:
        fq = parent_nodes[1]
        fq_name = make_copy_fake_quantize(nodes, edges, fq)
        edges.append((fq_name, node.name, {'out': 0, 'in': 1}))
    else:
        weights = parent_nodes[1]
        nodes.append((weights.name, weights.type, {'value': weights.value.copy()}))
        edges.append((weights.name, node.name, {'out': 0, 'in': 1}))

    if not remove_bias:
        if parent_nodes[2].type == 'FakeQuantize' and not remove_fake_quantize:
            fq = parent_nodes[1]
            fq_name = make_copy_fake_quantize(nodes, edges, fq)
            edges.append((fq_name, node.name, {'out': 0, 'in': 2}))
        else:
            weights = parent_nodes[2]
            nodes.append((weights.name, weights.type, {'value': weights.value.copy()}))
            edges.append((weights.name, node.name, {'out': 0, 'in': 2}))

    result_name = '{}/out'.format(node.name)
    nodes.append((result_name, 'Result', {}))
    edges.append((node.name, result_name, {'out': 0, 'in': 0}))
    graph = build_graph(*make_copy_graph_attrs(model, input_name, input_shape), nodes, edges)

    # Add the neccessary attribute to the new graph
    src_node = get_node_by_name(graph, node.name)
    weights_node = get_node_input(src_node, 1)
    weights_node = get_node_input(weights_node, 0) \
        if weights_node.type == 'FakeQuantize' else weights_node
    weights_out_dtype = weights_node.out_port(0).get_data_type()
    src_out_dtype = src_node.out_port(0).get_data_type()
    if weights_out_dtype != src_out_dtype:
        weights_node.out_node(0)['Insert_Convert_operation_after'] = True

    return graph
