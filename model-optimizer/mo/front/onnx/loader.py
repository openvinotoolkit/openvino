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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging as log

import networkx as nx
import onnx

from mo.graph.graph import create_graph_with_nodes, Graph
from mo.utils.error import Error, FrameworkError


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


def protobuf2nx(pb):
    '''Convert proto message with ONNX model to equivalent NX representation.
    All nodes and edges are restored here as ONNX model has op/data representation,
    that means that nodes are connected via tensor names. Name of tensors are defined
    on demand in nodes, so we have a code similar to Caffe here. '''
    # graph = create_graph_with_nodes(pb.graph.node, get_id=node_id, get_attrs=protobuf_attrs)
    # convert initializers to a NX graph for easier control of model consistency and to use it as a dictionary later
    initializers = create_graph_with_nodes(pb.graph.initializer, get_id=lambda pb: pb.name, get_attrs=protobuf_attrs)

    graph = Graph()

    # maps a tensor name to a node produced it and the node port: str -> (node_id, node_port)
    data_nodes_map = {}

    # first go through all inputs and separate constant from placeholders
    for inp in pb.graph.input:
        name = str(inp.name)
        if graph.has_node(name):
            raise Error('Name {} of input node already exists, input names are duplicated.', name)
        elif initializers.has_node(name):
            # this is a constant
            graph.add_node(name, kind='op', op='Const', pb=inp, pb_init=initializers.node[name]['pb'])
        else:
            # this is a placeholder
            graph.add_node(name, kind='op', op='Parameter', pb=inp)
        # add to a tensors map
        assert not name in data_nodes_map, 'Inconsistency between data_nodes_map and graph.nodes'
        data_nodes_map[name] = (name, 0)

    # go over all initializer and make sure that all of them are added to the graph
    for initializer in initializers.nodes():
        if not graph.has_node(initializer):
            graph.add_node(initializer, kind='op', op='Const', pb=initializers.node[initializer]['pb'],
                           pb_init=initializers.node[initializer]['pb'])
            data_nodes_map[initializer] = (initializer, 0)

    # Go through all nodes in the original model order (because data nodes are defined on-the-fly and order is
    # important)
    for node in pb.graph.node:
        # create an NX node
        id = graph.unique_id(node_id(node))
        graph.add_node(id, pb=node, kind='op')

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
                'fw_tensor_debug_info': [(inp, inp)],
                'in_attrs': ['in', 'name'],
                'out_attrs': ['out', 'name'],
                'data_attrs': ['fw_tensor_debug_info']
            }
            graph.add_edge(src_id, id, **edge_attrs)

        # add outgoing edges to data_nodes_map
        for src_port, out in enumerate(node.output):
            if out in data_nodes_map:
                log.debug("Detected reuse of blob {}.".format(out))
            data_nodes_map[out] = (id, src_port)

    return graph
