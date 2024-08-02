# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging as log

import onnx

from openvino.tools.mo.graph.graph import fill_graph_with_nodes, Graph, Node
from openvino.tools.mo.utils.error import Error, FrameworkError


def load_onnx_model(file_name: str):
    try:
        onnx_model = onnx.load(file_name)
    except Exception as e:
        raise FrameworkError(
            'Cannot read the model file: "{}" is incorrect ONNX model file. Details: {}',
            file_name,
            str(e)
        ) from e

    return onnx_model


def protobuf_attrs(pb):
    return {'pb': pb}


def node_id(pb):
    ''' The result of this function should be passed to unique_id to be used as a unuque ID for new node creation. '''
    if pb.name:
        return str(pb.name)
    elif len(pb.output):
        # node may have multiple outputs, we choose the first one
        return pb.output[0]
    else:
        return 'NoNamed'


def protobuf2nx(graph: Graph, pb):
    """
    Convert proto message with ONNX model to equivalent NX representation. All nodes and edges are restored here as
    ONNX model has op/data representation, that means that nodes are connected via tensor names. Name of tensors are
    defined on demand in nodes, so we have a code similar to Caffe here.

    :param graph: the Graph object to load the graph into
    :param pb: the ONNX file protobuf message
    :return: None
    """
    # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
    data_nodes_map = {}

    graph_pb = pb.graph
    add_initializers_and_inputs_to_graph(graph, graph_pb, data_nodes_map)

    # Preserve inputs order
    graph.inputs_order = []
    for inp in graph_pb.input:
        name = str(inp.name)
        graph.inputs_order.append(name)

    output_ids = []
    for outp in graph_pb.output:
        name = str(outp.name)
        if graph.has_node(name):
            log.error('Name {} of output node already exists in graph. Ignoring this output. If the output is required,'
                      ' please rename it.'.format(name), extra={'is_warning': True})
            continue
        else:
            # add fake node on output
            graph.add_node(name, kind='op', op='FakeOutput', pb=outp)
            output_ids.append(name)

    # Preserve outputs order
    graph.outputs_order = output_ids

    # Go through all nodes in the original model order (because data nodes are defined on-the-fly and order is
    # important)
    for node in graph_pb.node:
        # create an NX node
        fw_name = node_id(node)
        id = graph.unique_id(fw_name)
        graph.add_node(id, pb=node, kind='op')
        if hasattr(graph, 'op_names_statistic') and hasattr(node, 'op_type'):
            graph.op_names_statistic[node.op_type] += 1

        # add incoming edges based on data_nodes_map
        for dst_port, inp in enumerate(node.input):
            # should add edge inp --> id
            if inp not in data_nodes_map:
                if inp == '':
                    # input is omitted; most likely it corresponds to an optional input for an operator
                    continue
                else:
                    raise Error(
                        'Reference to {} is not satisfied. A node refer not existing data tensor. ONNX model is not '
                        'consistent. Protobuf fragment: {}', inp, node)
            src_id, src_port = data_nodes_map[inp]

            assert (graph.has_node(src_id))
            edge_attrs = {
                'out': src_port,
                'in': dst_port,
                'name': inp,
                'fw_tensor_debug_info': [(src_id, inp)],
                'in_attrs': ['in', 'name'],
                'out_attrs': ['out', 'name'],
                'data_attrs': ['fw_tensor_debug_info']
            }
            graph.add_edge(src_id, id, **edge_attrs)

        # add outgoing edges to data_nodes_map
        for src_port, out in enumerate(node.output):
            if out in output_ids:
                edge_attrs = {
                    'out': src_port,
                    'in': 0,
                    'name': out,
                    'fw_tensor_debug_info': [(fw_name, out)],
                    'in_attrs': ['in', 'name'],
                    'out_attrs': ['out', 'name'],
                    'data_attrs': ['fw_tensor_debug_info']
                }
                graph.add_edge(id, out, **edge_attrs)
            if out in data_nodes_map:
                log.debug("Detected reuse of blob {}.".format(out))
            data_nodes_map[out] = (id, src_port)

    graph.graph['tensor_mapping'] = data_nodes_map  # save main graph tensor names mapping for Loop op parsing


def add_initializers_and_inputs_to_graph(graph: Graph, graph_pb, data_nodes_map: dict):
    """
    The function adds nodes specified in the 'initializer' attribute of the pb and input nodes.
    :param graph: the Graph to add nodes to
    :param graph_pb: the graph protobuf message
    :param data_nodes_map: the dictionary with mapping of tensor names to node id and port
    :return: the list of Parameter nodes
    """
    initializers = Graph()
    fill_graph_with_nodes(initializers, graph_pb.initializer, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)

    parameters = []
    # first go through all inputs and separate constant from placeholders
    for inp in graph_pb.input:
        name = str(inp.name)
        if graph.has_node(name):
            raise Error('Name {} of input node already exists, input names are duplicated.', name)
        elif initializers.has_node(name):
            graph.add_node(name, kind='op', op='Const', pb=inp, pb_init=initializers.node[name]['pb'])
        else:
            graph.add_node(name, kind='op', op='Parameter', pb=inp)
            parameters.append(Node(graph, name))

        assert name not in data_nodes_map, 'Inconsistency between data_nodes_map and graph.nodes'
        data_nodes_map[name] = (name, 0)

    # go over all initializers and make sure that all of them are added to the graph
    for initializer in initializers.nodes():
        initializer_id = initializer
        if not graph.has_node(initializer_id):
            graph.add_node(initializer_id, kind='op', op='Const', pb=initializers.node[initializer]['pb'],
                           pb_init=initializers.node[initializer]['pb'])
            data_nodes_map[initializer] = (initializer_id, 0)
    return parameters
