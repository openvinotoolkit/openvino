# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
from io import IOBase

import networkx as nx
import numpy as np

from openvino.tools.mo.ops.elementwise import Mul
from openvino.tools.mo.ops.split import AttributedVariadicSplit
from openvino.tools.mo.front.common.partial_infer.utils import float_array, int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import add_outputs_identity
from openvino.tools.mo.front.kaldi.loader.utils import find_next_tag, read_placeholder, find_next_component, get_name_from_path, \
    find_end_of_component, end_of_nnet_tag, read_binary_integer32_token, get_parameters, read_token_value, \
    collect_until_token, collect_until_token_and_read, create_edge_attrs, get_args_for_specifier
from openvino.tools.mo.front.kaldi.utils import read_binary_vector
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


def load_parallel_component(file_descr, graph: Graph, prev_layer_id):
    """
    Load ParallelComponent of the Kaldi model.
    ParallelComponent contains parallel nested networks.
    VariadicSplit is inserted before nested networks.
    Outputs of nested networks concatenate with layer Concat.

    :param file_descr: descriptor of the model file
    :param graph: graph with the topology.
    :param prev_layer_id: id of the input layers for parallel component layer
    :return: id of the concat layer - last layer of the parallel component layers
    """
    nnet_count = read_token_value(file_descr, b'<NestedNnetCount>')
    log.debug('Model contains parallel component with {} nested networks'.format(nnet_count))

    split_points = []
    outputs = []
    inputs = []

    for i in range(nnet_count):
        read_token_value(file_descr, b'<NestedNnet>')
        collect_until_token(file_descr, b'<Nnet>')
        g = Graph()
        load_kalid_nnet1_model(g, file_descr, 'Nested_net_{}'.format(i))

        # input to nnet1 models is of a rank 1 but we also insert batch_size to 0th axis
        # 1st axis contains input_size of the nested subnetwork
        # we split input from the main network to subnetworks
        input_node = Node(g, 'Parameter')
        split_points.append(input_node['shape'][1])
        g.remove_node(input_node.id)

        mapping = {node: graph.unique_id(node) for node in g.nodes(data=False) if node in graph}
        g = nx.relabel_nodes(g, mapping)
        for val in mapping.values():
            g.node[val]['name'] = val
        graph.add_nodes_from(g.nodes(data=True))
        graph.add_edges_from(g.edges(data=True))
        sorted_nodes = tuple(nx.topological_sort(g))

        outputs.append(Node(graph, sorted_nodes[-1]))
        inputs.append(Node(graph, sorted_nodes[0]))

    split_id = graph.unique_id(prefix='NestedNets/VariadicSplit')
    attrs = {'out_ports_count': nnet_count, 'size_splits': split_points, 'axis': 1, 'name': split_id}
    variadic_split_node = AttributedVariadicSplit(graph, attrs).create_node()
    prev_layer_node = Node(graph, prev_layer_id)
    prev_layer_node.add_output_port(0)
    graph.create_edge(prev_layer_node, variadic_split_node, 0, 0, create_edge_attrs(prev_layer_id, variadic_split_node.id, prev_layer_id))

    concat_id = graph.unique_id(prefix='Concat')
    graph.add_node(concat_id, parameters=None, op='concat', kind='op')
    concat_node = Node(graph, concat_id)

    # Connect each output of variadic_split_node to each subnetwork's inputs in ParallelComponent
    # and each subnetwork's output to concat_node
    for i, (input_node, output_node) in enumerate(zip(inputs, outputs)):
        output_node.add_output_port(0)
        concat_node.add_input_port(i)
        graph.create_edge(output_node, concat_node, 0, i, create_edge_attrs(output_node.id, concat_id, output_node.id, i, 0))
        graph.create_edge(variadic_split_node, input_node, i, 0, create_edge_attrs(variadic_split_node.id, input_node.id, variadic_split_node.id, 0, i))
    return concat_id


def load_kaldi_model(graph, nnet_path):
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

    tag = find_next_tag(file_desc)
    # start new model / submodel
    if tag == '<Nnet>':
        load_function = load_kalid_nnet1_model
    elif tag == '<TransitionModel>':
        while tag != '<Nnet>' and tag != '<Nnet3>':
            tag = find_next_tag(file_desc)

        if tag == '<Nnet3>':
            load_function = load_kaldi_nnet3_model
        else:
            load_function = load_kalid_nnet2_model
    elif tag == '<Nnet3>':
        load_function = load_kaldi_nnet3_model
    else:
        raise Error('Kaldi model should start with <Nnet> or <TransitionModel> tag. ',
                    refer_to_faq_msg(89))
    read_placeholder(file_desc, 1)

    return load_function(graph, file_desc, nnet_name)


def load_kalid_nnet1_model(graph, file_descr, name):
    prev_layer_id = 'Parameter'
    graph.add_node(prev_layer_id, name=prev_layer_id, kind='op', op='Parameter', parameters=None)

    # find out output layer, it can be only one due to chain structure of nnet1 model
    output_layer = None
    while True:
        component_type = find_next_component(file_descr)
        if component_type == end_of_nnet_tag.lower()[1:-1]:
            break

        layer_o = read_binary_integer32_token(file_descr)
        layer_i = read_binary_integer32_token(file_descr)

        if component_type == 'parallelcomponent':
            prev_layer_id = load_parallel_component(file_descr, graph, prev_layer_id)
            find_end_of_component(file_descr, component_type)
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
        if hasattr(graph, 'op_names_statistic'):
            graph.op_names_statistic[component_type] += 1

        prev_node = Node(graph, prev_layer_id)
        if prev_node.op == 'Parameter':
            prev_node['shape'] = int64_array([1, layer_i])

        prev_node.add_output_port(0)
        Node(graph, layer_id).add_input_port(0)
        graph.create_edge(prev_node, Node(graph, layer_id), 0, 0, create_edge_attrs(prev_layer_id, layer_id, prev_layer_id))
        prev_layer_id = layer_id
        output_layer = layer_id
        log.debug('{} (type is {}) was loaded'.format(prev_layer_id, component_type))

    # Tensor names information corresponding to a node is stored on outgoing edges.
    # As output nodes do not have outgoing edges, fake outputs are required. In the following code
    # for each output Identity node is added, and tensor name for the output is kept
    # on (output, fake output) edge. After Result nodes adding transformation fake outputs
    # are deleted from graph.
    assert output_layer is not None, "Output layer is not found in graph"
    add_outputs_identity(graph, [output_layer], lambda g, output, fake_output: g.create_edge(
        Node(g, output), Node(g, fake_output), 0, 0, create_edge_attrs(output, fake_output, output)))


def load_kalid_nnet2_model(graph, file_descr, nnet_name):
    input_name = 'Input'
    graph.add_node(input_name, name=input_name, kind='op', op='Parameter', parameters=None, shape=None)

    prev_layer_id = input_name

    all_components = load_components(file_descr, graph)

    used_layers = set()
    for layer_id in all_components:
        prev_node = Node(graph, prev_layer_id)
        if prev_node.op == 'Parameter':
            parameters = Node(graph, layer_id).parameters
            input_dim = read_token_value(parameters, b'<InputDim>')
            prev_node['shape'] = int64_array([1, input_dim])
        prev_node.add_output_port(0)
        Node(graph, layer_id).add_input_port(0)
        graph.create_edge(prev_node, Node(graph, layer_id), 0, 0, create_edge_attrs(prev_layer_id, layer_id, prev_layer_id))
        used_layers.add(prev_layer_id)
        prev_layer_id = layer_id
        log.debug('{} and {} were connected'.format(prev_layer_id, layer_id))

    # Tensor names information corresponding to a node is stored on outgoing edges.
    # As output nodes do not have outgoing edges, fake outputs are required. In the following code
    # for each output Identity node is added, and tensor name for the output is kept
    # on (output, fake output) edge. After Result nodes adding transformation fake outputs
    # are deleted from graph.
    output_layers = graph.nodes - used_layers
    add_outputs_identity(graph, output_layers, lambda g, output, fake_output: g.create_edge(
        Node(g, output), Node(g, fake_output), 0, 0, create_edge_attrs(output, fake_output, output)))


def load_kaldi_nnet3_model(graph, file_descr, nnet_name):
    file_descr.read(1)
    component_layer_map = load_topology_map(file_descr, graph)
    # add information for shape calculation for MemoryOffset
    # shape calculation for MemoryOffset can't be done through shape of previous layer because
    # it is separated in 2 parts to remove cycle from graph
    for node in graph.get_op_nodes(**{'op': 'Parameter'}):
        for o_n_name, params in node.get_outputs():
            o_n = Node(graph, o_n_name)
            if o_n['op'] == 'MemoryOffset':
                # don't take batch from Parameter, it will be overwritten
                # take only second dimension because we have only 2 dimensions
                o_n['parameters']['element_size'] = int64_array([1, node.shape[1]])

    load_components(file_descr, graph, component_layer_map)
    load_priors(file_descr, graph)


def load_priors(file_descr, graph):
    try:
        collect_until_token(file_descr, b'<Priors>')
    except Error:
        # just ignore if priors were not found
        return
    if graph.graph['cmd_params'].counts is not None:
        graph.graph['priors'] = read_binary_vector(file_descr)
    else:
        log.error("Model contains Prior values, if you want to embed them into the generated IR add option --counts=\"\" to command line",
                  extra={'is_warning': True})


def load_components(file_descr, graph, component_layer_map=None):
    num_components = collect_until_token_and_read(file_descr, b'<NumComponents>')
    log.debug('Network contains {} components'.format(num_components))
    is_nnet3 = False if component_layer_map is None else True

    if not is_nnet3:
        collect_until_token(file_descr, b'<Components>')

    all_components = list()
    name = ""
    for _ in range(num_components):
        if is_nnet3:
            name = collect_until_token_and_read(file_descr, b'<ComponentName>', np.string_)

        component_type = find_next_component(file_descr)
        if component_type == end_of_nnet_tag.lower()[1:-1]:
            break

        start_index = file_descr.tell()
        end_tag, end_index = find_end_of_component(file_descr, component_type)
        # read dim info where possible to simplify shape calculation for MemoryOffset
        # shape calculation for MemoryOffset can't be done through shape of previous layer because
        # it is separated in 2 parts to remove cycle from graph
        file_descr.seek(start_index)
        dim = 0
        dim_words = {b'<Dim>', b'<InputDim>'}
        for dim_word in dim_words:
            try:
                collect_until_token(file_descr, dim_word, size_search_zone=end_index - start_index)
                cur_index = file_descr.tell()
                if start_index < cur_index < end_index:
                    dim = read_binary_integer32_token(file_descr)
                    break
                else:
                    file_descr.seek(start_index)
            except Error:
                file_descr.seek(start_index)

        if is_nnet3:
            if name in component_layer_map:
                layer_id = component_layer_map[name][0]
                for layer in component_layer_map[name]:
                    node = Node(graph, layer)
                    node['parameters'] = get_parameters(file_descr, start_index, end_index)
                    node['op'] = component_type
                    # Read dim info where possible to simplify shape calculation for MemoryOffset
                    for o_n_name, params in node.get_outputs():
                        o_n = Node(graph, o_n_name)
                        if o_n['op'] == 'MemoryOffset' and dim != 0:
                            o_n['parameters']['element_size'] = int64_array([1, dim])
            else:
                raise Error("Something wrong with layer {}".format(name))
        else:
            layer_id = graph.unique_id(prefix=component_type)
            graph.add_node(layer_id,
                           parameters=get_parameters(file_descr, start_index, end_index),
                           op=component_type,
                           kind='op')
        if hasattr(graph, 'op_names_statistic'):
            graph.op_names_statistic[component_type] += 1

        all_components.append(layer_id)
        log.debug('{} (type is {}) was loaded'.format(layer_id, component_type))

    return all_components


def load_topology_map(file_descr, graph):
    not_finished = True
    component_layer_map = {}
    layer_node_map = {}
    while not_finished:
        not_finished = read_node(file_descr, graph, component_layer_map, layer_node_map)
    return component_layer_map


def read_node(file_descr, graph, component_layer_map, layer_node_map):
    s = file_descr.readline()
    if s == b'\n':
        return False
    tokens = s.split(b' ')
    if tokens[0] == b'input-node':
        in_name = s[s.find(b'name=') + len(b'name='):].split(b' ')[0]
        in_name = str(in_name).strip('b').replace('\'', "")
        in_shape = mo_array([1, s[s.find(b'dim=') + len(b'dim='):].split(b' ')[0]], dtype=int)

        if in_name not in layer_node_map:
            graph.add_node(in_name, name=in_name, kind='op', op='Parameter', parameters=None, shape=in_shape)
            layer_node_map[in_name] = in_name
        else:
            Node(graph, in_name)['op'] = 'Parameter'
            Node(graph, in_name)['shape'] = in_shape
    elif tokens[0] == b'component-node':
        layer_name = s[s.find(b'name=') + len(b'name='):].split(b' ')[0]
        layer_name = str(layer_name).strip('b').replace('\'', "")

        component_name = s[s.find(b'component=') + len(b'component='):].split(b' ')[0]
        if layer_name not in layer_node_map:
            node_name = graph.unique_id(prefix=layer_name)
            graph.add_node(node_name,
                           parameters=None,
                           op=None,
                           kind='op')
            layer_node_map[layer_name] = node_name
        else:
            node_name = layer_node_map[layer_name]

        if component_name in component_layer_map:
            component_layer_map[component_name].append(node_name)
        else:
            component_layer_map[component_name] = [node_name]

        # parse input
        in_node_id = parse_input_for_node(s[s.find(b'input=') + 6:], graph, layer_node_map)
        # don't create cyclic edges node to itself to avoid removing later
        if in_node_id != node_name:
            out_port = len(Node(graph, in_node_id).out_nodes())
            in_port = len(Node(graph, node_name).in_nodes())

            Node(graph, node_name).add_input_port(in_port)
            Node(graph, in_node_id).add_output_port(out_port, skip_if_exist=True)

            graph.add_edge(in_node_id, node_name, **create_edge_attrs(in_node_id, node_name, in_node_id, in_port, out_port))
    elif tokens[0] == b'output-node':
        layer_name = s[s.find(b'name=') + len(b'name='):].split(b' ')[0]
        layer_name = str(layer_name).strip('b').replace('\'', "")
        node_name = graph.unique_id(prefix=layer_name)
        graph.add_node(node_name,
                       parameters=None,
                       op='Identity',
                       kind='op')
        out_name = graph.unique_id(prefix=node_name + "_out")
        graph.add_node(out_name,
                       parameters=None,
                       op='Result',
                       kind='op')
        Node(graph, node_name).add_input_port(0)
        Node(graph, node_name).add_output_port(0)
        Node(graph, out_name).add_input_port(0)
        graph.add_edge(node_name, out_name, **create_edge_attrs(node_name, out_name, node_name))

        # parse input
        in_node_id = parse_input_for_node(s[s.find(b'input=') + len(b'input='):], graph, layer_node_map)

        out_port = len(Node(graph, in_node_id).out_nodes())
        Node(graph, in_node_id).add_output_port(out_port)
        graph.create_edge(Node(graph, in_node_id), Node(graph, node_name), out_port, 0, create_edge_attrs(in_node_id, node_name, in_node_id, 0, out_port))

        objective_type = s[s.find(b'objective=') + 10:].split(b' ')[0].split(b'\n')[0]
        if objective_type != b'linear':
            raise Error("Unsupported objective-type for output {}".format(node_name))
    elif tokens[0] == b'dim-range-node':
        layer_name = s[s.find(b'name=') + len(b'name='):].split(b' ')[0]
        layer_name = str(layer_name).strip('b').replace('\'', "")
        offset = int(s[s.find(b'dim-offset=') + len(b'dim-offset='):].split(b' ')[0])
        dim = int(s[s.find(b'dim=') + len(b'dim='):].split(b' ')[0])

        if layer_name in layer_node_map:
            node_name = layer_node_map[layer_name]
            node = Node(graph, node_name)
            node['parameters'] = {'offset': mo_array([offset]), 'dim': mo_array([dim]), 'axis': mo_array([1])}
            node['op'] = 'Crop'
        else:
            node_name = graph.unique_id(prefix=layer_name)
            graph.add_node(node_name,
                           parameters={'offset': mo_array([offset]), 'dim': mo_array([dim]), 'axis': mo_array([1])},
                           op='Crop',
                           kind='op')
            layer_node_map[layer_name] = node_name
            node = Node(graph, node_name)

        in_node_id = parse_input_for_node(s[s.find(b'input-node=') + len(b'input-node='):], graph, layer_node_map)
        out_port = len(Node(graph, in_node_id).out_nodes())
        in_port = len(Node(graph, node_name).in_nodes())

        node.add_input_port(in_port)
        Node(graph, in_node_id).add_output_port(out_port)

        graph.create_edge(Node(graph, in_node_id), node, out_port, in_port, create_edge_attrs(in_node_id, node_name, in_node_id, in_port, out_port))

        # read dim info where possible to simplify shape calculation for MemoryOffset
        # shape calculation for MemoryOffset can't be done through shape of previous layer because
        # it is separated in 2 parts to remove cycle from graph
        for o_n_name, params in node.get_outputs():
            o_n = Node(graph, o_n_name)
            if o_n['op'] == 'MemoryOffset':
                o_n['parameters']['element_size'] = int64_array([1, dim])
    else:
        raise Error("Unsupported node specifier {}".format(tokens[0]))
    return True


def parse_input_for_node(string, graph, component_layer_map):
    return parse_specifier(string, graph, component_layer_map)


def parse_specifier(string, graph, layer_node_map):
    pos = string.find(b'(')
    if pos == -1:
        # node name
        input_name = str(string.split(b' ')[0]).strip('b').replace("\'", '').replace('\\n', '')

        if input_name not in layer_node_map:
            node_name = graph.unique_id(prefix=input_name)
            graph.add_node(node_name, parameters=[], op="", kind='op')
            layer_node_map[input_name] = node_name
        else:
            node_name = layer_node_map[input_name]
        return node_name

    spec = string[:pos]
    args = get_args_for_specifier(string[pos:])
    if spec == b'Append':
        nodes = []
        for i in range(len(args)):
            nodes.append(parse_specifier(args[i], graph, layer_node_map))
        layer_name = 'Append_'
        for node in nodes:
            layer_name = layer_name + node + "_"

        if layer_name not in layer_node_map:
            concat_name = graph.unique_id(prefix=layer_name)
            graph.add_node(concat_name,
                           parameters=None,
                           op='concat',
                           kind='op')
            layer_node_map[layer_name] = concat_name
            i = 0
            Node(graph, concat_name).add_sequence_of_ports('in', range(len(nodes)))
            for node in nodes:
                out_port = len(Node(graph, node).out_nodes())
                Node(graph, node).add_output_port(out_port)
                graph.create_edge(Node(graph, node), Node(graph, concat_name), out_port, i, create_edge_attrs(node, concat_name, node, i, out_port))
                i = i + 1
        else:
            concat_name = layer_node_map[layer_name]
        return concat_name
    elif spec == b'Offset':
        node = parse_specifier(args[0], graph, layer_node_map)
        t = int(args[1])
        if len(args) > 2:
            raise Error("ModelOptimizer supports only 2 arguments for Offset")
        layer_name = 'Offset_' + node + '_'
        if t < 0:
            layer_name = layer_name + '_' + str(-t)
        else:
            layer_name = layer_name + str(t)

        if layer_name not in layer_node_map:
            memory_name = graph.unique_id(prefix=layer_name)
            layer_node_map[layer_name] = memory_name
            memory_name_2 = memory_name + '_out'
            graph.add_node(memory_name,
                           parameters=dict(t=t, pair_name=memory_name_2, has_default=False),
                           op='MemoryOffset',
                           kind='op')
            out_port = len(Node(graph, node).out_nodes())
            in_port = len(Node(graph, memory_name).in_nodes())
            Node(graph, memory_name).add_input_port(in_port)
            Node(graph, node).add_output_port(out_port, skip_if_exist=True)
            graph.create_edge(Node(graph, node), Node(graph, memory_name), out_port, in_port, create_edge_attrs(node, memory_name, node, in_port, out_port))
        else:
            memory_name = layer_node_map[layer_name]
        return memory_name
    elif spec == b'Sum':
        nodes = []
        for i in range(len(args)):
            nodes.append(parse_specifier(args[i], graph, layer_node_map))

        layer_name = 'Sum_'
        for node in nodes:
            layer_name = layer_name + node + "_"

        if layer_name not in layer_node_map:
            sum_name = graph.unique_id(prefix=layer_name)
            graph.add_node(sum_name, parameters=None, op='Add', kind='op')
            layer_node_map[layer_name] = sum_name
        else:
            sum_name = layer_node_map[layer_name]

        for i, node in enumerate(nodes):
            out_port = len(Node(graph, node).out_nodes())
            Node(graph, node).add_output_port(out_port, skip_if_exist=True)
            Node(graph, sum_name).add_input_port(i)
            graph.add_edge(node, sum_name, **create_edge_attrs(node, sum_name, node, i))

        return sum_name
    elif spec == b'IfDefined':
        node_id = parse_specifier(args[0], graph, layer_node_map)
        node = Node(graph, node_id)
        if node.op == 'MemoryOffset':
            node['parameters']['has_default'] = True
        return node_id
    elif spec == b'ReplaceIndex':
        node = parse_specifier(args[0], graph, layer_node_map)
        return node
    elif spec == b'Scale':
        node_name = parse_specifier(args[1], graph, layer_node_map)
        scale_value = float(args[0])
        layer_name = '{}/Mul/{}'.format(node_name, scale_value)

        if layer_name not in layer_node_map:
            scale_name = graph.unique_id(prefix=layer_name)
            scale_node = Mul(graph, {'name': scale_name}).create_node()

            layer_node_map[layer_name] = scale_name

            scale_const_name = 'Const_{}'.format(scale_value)
            const_node = Const(graph, {'name': scale_const_name, 'value': float_array([scale_value])}).create_node()

            node = Node(graph, node_name)
            graph.create_edge(const_node, scale_node, 0, 0, create_edge_attrs(const_node.id, scale_node.id, const_node.id))
            out_port = len(node.out_nodes())
            graph.create_edge(node, scale_node, out_port, 1, create_edge_attrs(node_name, scale_node.id, node_name, 1, out_port))
        else:
            scale_name = layer_node_map[layer_name]

        return scale_name
