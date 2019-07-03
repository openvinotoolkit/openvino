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
import io

import numpy as np
import struct
from io import IOBase

import networkx as nx
import logging as log

from mo.front.kaldi.loader.utils import find_next_tag, read_placeholder, find_next_component, get_name_from_path, \
    find_end_of_component, end_of_nnet_tag, read_binary_integer32_token, get_parameters, read_token_value, collect_until_token, \
    create_edge_attrs
from mo.graph.graph import Node, Graph
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def read_counts_file(file_path):
    with open(file_path, 'r') as f:
        file_content = f.readlines()
    if len(file_content) > 1:
        raise Error('Expect counts file to be one-line file. ' +
                    refer_to_faq_msg(90))

    counts_line = file_content[0].strip().replace('[', '').replace(']', '')
    try:
        counts = np.fromstring(counts_line, dtype=float, sep=' ')
    except TypeError:
        raise Error('Expect counts file to contain list of integers.' +
                    refer_to_faq_msg(90))
    cutoff = 1.00000001e-10
    cutoff_idxs = np.where(counts < cutoff)
    counts[cutoff_idxs] = cutoff
    scale = 1.0 / np.sum(counts)
    counts = np.log(counts * scale)  # pylint: disable=assignment-from-no-return
    counts[cutoff_idxs] += np.finfo(np.float32).max / 2
    return counts


def load_parallel_component(file_descr, graph: Graph, prev_layer_id):
    """
    Load ParallelComponent of the Kaldi model.
    ParallelComponent contains parallel nested networks.
    Slice is inserted before nested networks.
    Outputs of nested networks concatenate with layer Concat.

    :param file_descr: descriptor of the model file
    :param graph: graph with the topology.
    :param prev_layer_id: id of the input layers for parallel component layer
    :return: id of the concat layer - last layer of the parallel component layers
    """
    nnet_count = read_token_value(file_descr, b'<NestedNnetCount>')
    log.debug('Model contains parallel component with {} nested networks'.format(nnet_count))

    slice_id = graph.unique_id(prefix='Slice')
    graph.add_node(slice_id, parameters=None, op='slice', kind='op')

    slice_node = Node(graph, slice_id)
    graph.add_edge(prev_layer_id, slice_id, **create_edge_attrs(prev_layer_id, slice_id))
    slices_points = []

    outputs = []

    for i in range(nnet_count):
        read_token_value(file_descr, b'<NestedNnet>')
        collect_until_token(file_descr, b'<Nnet>')
        g, shape = load_kalid_nnet1_model(file_descr, 'Nested_net_{}'.format(i))
        input_nodes = [n for n in graph.nodes(data=True) if n[1]['op'] == 'Input']
        if i != nnet_count - 1:
            slices_points.append(shape[1])
        g.remove_node(input_nodes[0][0])
        mapping = {node: graph.unique_id(node) for node in g.nodes(data=False) if node in graph}
        g = nx.relabel_nodes(g, mapping)
        for val in mapping.values():
            g.node[val]['name'] = val
        graph.add_nodes_from(g.nodes(data=True))
        graph.add_edges_from(g.edges(data=True))
        sorted_nodes = tuple(nx.topological_sort(g))
        edge_attrs = create_edge_attrs(slice_id, sorted_nodes[0])
        edge_attrs['out'] = i
        graph.add_edge(slice_id, sorted_nodes[0], **edge_attrs)
        outputs.append(sorted_nodes[-1])
    packed_sp = struct.pack("B", 4) + struct.pack("I", len(slices_points))
    for i in slices_points:
        packed_sp += struct.pack("I", i)
    slice_node.parameters = io.BytesIO(packed_sp)
    concat_id = graph.unique_id(prefix='Concat')
    graph.add_node(concat_id, parameters=None, op='concat', kind='op')
    for i, output in enumerate(outputs):
        edge_attrs = create_edge_attrs(output, concat_id)
        edge_attrs['in'] = i
        graph.add_edge(output, concat_id, **edge_attrs)
    return concat_id


def load_kaldi_model(nnet_path):
    """
    Structure of the file is the following:
    magic-number(16896)<Nnet> <Next Layer Name> weights etc.
    :param nnet_path:
    :return:
    """
    nnet_name = None
    if isinstance(nnet_path, str):
        file_desc = open(nnet_path, "rb")
        nnet_name = get_name_from_path(nnet_path)
    elif isinstance(nnet_path, IOBase):
        file_desc = nnet_path
    else:
        raise Error('Unsupported type of Kaldi model')

    name = find_next_tag(file_desc)
    # start new model / submodel
    if name == '<Nnet>':
        load_function = load_kalid_nnet1_model
    elif name == '<TransitionModel>':
        load_function = load_kalid_nnet2_model
    else:
        raise Error('Kaldi model should start with <Nnet> or <TransitionModel> tag. ',
                    refer_to_faq_msg(89))
    read_placeholder(file_desc, 1)

    return load_function(file_desc, nnet_name)


def load_kalid_nnet1_model(file_descr, name):
    graph = Graph(name=name)

    prev_layer_id = 'Input'
    graph.add_node(prev_layer_id, name=prev_layer_id, kind='op', op='Input', parameters=None)
    input_shape = []

    while True:
        component_type = find_next_component(file_descr)
        if component_type == end_of_nnet_tag.lower()[1:-1]:
            break

        layer_o = read_binary_integer32_token(file_descr)
        layer_i = read_binary_integer32_token(file_descr)

        if component_type == 'parallelcomponent':
            prev_layer_id = load_parallel_component(file_descr, graph, prev_layer_id)
            continue

        start_index = file_descr.tell()
        end_tag, end_index = find_end_of_component(file_descr, component_type)
        end_index -= len(end_tag)
        layer_id = graph.unique_id(prefix=component_type)
        graph.add_node(layer_id,
                       parameters=get_parameters(file_descr, start_index, end_index),
                       op=component_type,
                       kind='op',
                       layer_i=layer_i,
                       layer_o=layer_o)

        prev_node = Node(graph, prev_layer_id)
        if prev_node.op == 'Input':
            prev_node['shape'] = np.array([1, layer_i], dtype=np.int64)
            input_shape = np.array([1, layer_i], dtype=np.int64)
        graph.add_edge(prev_layer_id, layer_id, **create_edge_attrs(prev_layer_id, layer_id))
        prev_layer_id = layer_id
        log.debug('{} (type is {}) was loaded'.format(prev_layer_id, component_type))
    return graph, input_shape


def load_kalid_nnet2_model(file_descr, nnet_name):
    graph = Graph(name=nnet_name)
    input_name = 'Input'
    input_shape = np.array([])
    graph.add_node(input_name, name=input_name, kind='op', op='Input', parameters=None, shape=None)

    prev_layer_id = input_name

    collect_until_token(file_descr, b'<Nnet>')
    num_components = read_token_value(file_descr, b'<NumComponents>')
    log.debug('Network contains {} components'.format(num_components))
    collect_until_token(file_descr, b'<Components>')
    for _ in range(num_components):
        component_type = find_next_component(file_descr)

        if component_type == end_of_nnet_tag.lower()[1:-1]:
            break
        start_index = file_descr.tell()
        end_tag, end_index = find_end_of_component(file_descr, component_type)
        layer_id = graph.unique_id(prefix=component_type)
        graph.add_node(layer_id,
                       parameters=get_parameters(file_descr, start_index, end_index),
                       op=component_type,
                       kind='op')

        prev_node = Node(graph, prev_layer_id)
        if prev_node.op == 'Input':
            parameters = Node(graph, layer_id).parameters
            input_dim = read_token_value(parameters, b'<InputDim>')
            prev_node['shape'] = np.array([1, input_dim], dtype=np.int64)
            input_shape = np.array([1, input_dim], dtype=np.int64)
        graph.add_edge(prev_layer_id, layer_id, **create_edge_attrs(prev_layer_id, layer_id))
        prev_layer_id = layer_id
        log.debug('{} (type is {}) was loaded'.format(prev_layer_id, component_type))
    return graph, input_shape
