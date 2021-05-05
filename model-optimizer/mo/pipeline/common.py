# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import os
from operator import itemgetter

import networkx as nx

from extensions.back.RemoveUselessConvert import RemoveUselessConvert
from extensions.back.ResultRename import ResultRename
from extensions.back.op_versioning import OpVersioning
from extensions.ops.Cast import Cast
from mo.back.ie_ir_ver_2.emitter import port_renumber, serialize_constants, generate_ie_ir, serialize_mean_image
from mo.graph.graph import Node, Graph
from mo.middle.passes import tensor_names, convert_data_type
from mo.middle.passes.convert_data_type import data_type_str_to_np
from mo.middle.passes.infer import type_infer
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.utils.error import Error


def determined_sort(outputs: list):
    op_order = []
    data_order = []
    stack = list(outputs)
    visited = set()
    while len(stack) != 0:
        node = stack.pop(0)
        node_id = node.id
        visited.add(node_id)
        has_child = False
        in_names = [n for n, d in node.get_inputs()]
        for in_node_name in in_names:
            if in_node_name not in visited:
                stack.insert(0, node)
                stack.insert(0, Node(node.graph, in_node_name))
                has_child = True
                break
        if not has_child:
            if node.kind == 'op':
                op_order.append(node_id)
            if node.kind == 'data':
                data_order.append(node_id)
    return op_order, data_order


def get_fw_tensor_debug_info(node: Node):
    while not node.has_valid('fw_tensor_debug_info') and not node.has_valid('output_sort_order') \
            and len(node.in_nodes()):
        try:
            node = node.in_node()
        except Exception as e:
            log.warning('Was not able to determine tensor debug info for node {}'.format(node.name))
            return "dummy_node_name"
    if node.has_valid('output_sort_order'):
        return node.soft_get('output_sort_order')
    return node.soft_get('fw_tensor_debug_info')


def get_sorted_outputs(graph: Graph):
    outputs = []
    outputs_for_sort = {}
    for node in graph.nodes():
        if len(graph.out_edges(node)) == 0:
            outputs.append(Node(graph, node))
    if len(outputs) == 1:
        return outputs
    for node in outputs:
        debug_info = get_fw_tensor_debug_info(node)
        if isinstance(debug_info, str):
            outputs_for_sort[node.id] = debug_info
        elif isinstance(debug_info, list):
            outputs_for_sort[node.id] = debug_info[0][0] + '_' + str(debug_info[0][1])
        else:
            raise Error('Unsupported type of the variable with debug information used to sort output nodes')
    if len(outputs_for_sort) != len(set(outputs_for_sort.values())):
        log.warning('There are at least two output nodes with the same key used to sort the outputs. This means that '
                    'IRs with different order of nodes may be generated between Model Optimizer runs. The dictionary '
                    'with outputs is: {}'.format(outputs_for_sort))
    return [Node(graph, key) for key, value in sorted(outputs_for_sort.items(), key=itemgetter(1))]


def collect_sub_graphs(graph: Graph):
    """ Go over all nodes and sub_graphs in the graph recursively; returns all found sub-graphs. """
    result = []
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('sub_graphs'):
            for sub_graph in node.sub_graphs:
                result.append(node[sub_graph])
                result += collect_sub_graphs(node[sub_graph])
    return result


def relabel_nodes_inplace_safe(graph: Graph, new_labels: dict):
    """ Safely relabels graph in-place without graph copy.
        
        Safety in this place means that it is guaranteed that
        there won't be collisions during relabeling process.
    """
    # Relabel nodes in two stages
    intermediate_map = {node: graph.unique_id('__relabel__{}__'.format(str(i))) for i, node in enumerate(graph.nodes())}
    final_map = {dst: new_labels[src] for src, dst in intermediate_map.items()}
    assert len(set(intermediate_map.keys()).intersection(set(intermediate_map.values()))) == 0
    assert len(set(final_map.keys()).intersection(set(final_map.values()))) == 0
    nx.relabel_nodes(graph, intermediate_map, copy=False)
    nx.relabel_nodes(graph, final_map, copy=False)


def convert_const_node_value_type(const_node: Node, np_data_type):
    assert const_node.type == 'Const'
    log.warning('Converting type of Const node "{}" to "{}"'.format(const_node.name, np_data_type))
    const_node.value = const_node.value.astype(np_data_type)
    const_node.data_type = np_data_type
    const_node.infer(const_node)
    const_node.type_infer(const_node)

    # if the Const node has an input data node then need to update it also
    if len(const_node.in_nodes()) == 1:
        input_data = const_node.in_node(0)
        assert input_data.kind == 'data'
        input_data.value = input_data.value.astype(const_node.data_type)
        input_data.data_type = const_node.data_type


def convert_inputs_of_specific_ops(graph: Graph):
    type_port = {'Broadcast': {1: 'int64', 2: 'int64'},
                 'ConvolutionBackpropData': {2: 'int64'},
                 'Deconvolution': {2: 'int64'},
                 'Gather': {2: 'int64'},
                 'GroupConvolutionBackpropData': {2: 'int64'},
                 'Interpolate': {1: 'int64'},
                 'LRN': {1: 'int64'},
                 'NonMaxSuppression': {2: 'int64'},
                 'NormalizeL2': {1: 'int64'},
                 'OneHot': {1: 'int64'},
                 'Pad': {1: 'int64', 2: 'int64'},
                 'PriorBox': {0: 'int64', 1: 'int64'},
                 'PriorBoxClustered': {0: 'int64', 1: 'int64'},
                 'ReduceLogicalAnd': {1: 'int64'},
                 'ReduceLogicalOr': {1: 'int64'},
                 'ReduceMax': {1: 'int64'},
                 'ReduceMean': {1: 'int64'},
                 'ReduceMin': {1: 'int64'},
                 'ReduceProd': {1: 'int64'},
                 'ReduceSum': {1: 'int64'},
                 'Reshape': {1: 'int64'},
                 'Squeeze': {1: 'int64'},
                 'StridedSlice': {1: 'int64', 2: 'int64', 3: 'int64'},
                 'Split': {1: 'int64'},
                 'Tile': {1: 'int64'},
                 'Transpose': {1: 'int64'},
                 'Unsqueeze': {1: 'int64'},
                 'VariadicSplit': {1: 'int64', 2: 'int64'},
                 }

    for node in graph.get_op_nodes():
        if node.soft_get('type') in type_port:
            ports_to_update = type_port[node.soft_get('type')]
            for port_id, precision in ports_to_update.items():
                if port_id in node.in_ports() and not node.in_port(port_id).disconnected():
                    log.debug('Converting value for the input port "{}" of op "{}" to "{}".'
                              ''.format(port_id, node.soft_get('name', node.id), precision))
                    in_port = node.in_port(port_id)
                    np_type = data_type_str_to_np(precision)
                    if in_port.get_source().node.type == 'Const':
                        convert_const_node_value_type(node.in_port(port_id).get_source().node, np_type)
                    else:
                        in_port.get_connection().insert_node(Cast(graph, {'dst_type': np_type}).create_node())


def prepare_emit_ir(graph: Graph, data_type: str, output_dir: str, output_model_name: str,
                    mean_data: [list, None] = None, input_names: list = None, meta_info: dict = None,
                    use_temporary_path=False):
    if input_names is None:
        input_names = []
    if meta_info is None:
        meta_info = {}
    graph.strict_mode = False

    # convert Parameter data types
    convert_data_type.convert_parameters_data_type(graph, data_type)
    # convert blobs (usually weights and biases)
    for sub_graph in [graph] + collect_sub_graphs(graph):
        convert_data_type.convert_blobs(sub_graph, data_type)

    # restore data type for specific inputs/outputs of specific ops to the data types required by nGraph
    for_graph_and_each_sub_graph_recursively(graph, convert_inputs_of_specific_ops)

    for_graph_and_each_sub_graph_recursively(graph, OpVersioning().find_and_replace_pattern)

    # do not run the type inference in sub-graphs. It will be called automatically as part of the type inference of
    # the TensorIterator nodes
    type_infer(graph)
    RemoveUselessConvert().find_and_replace_pattern(graph)

    ResultRename().find_and_replace_pattern(graph)

    for sub_graph in [graph] + collect_sub_graphs(graph):
        op_order, data_order = determined_sort(get_sorted_outputs(sub_graph))
        mapping = {v: u for u, v in enumerate(op_order)}
        mapping.update({v: u for u, v in enumerate(data_order, start=len(sub_graph))})
        relabel_nodes_inplace_safe(sub_graph, mapping)
        port_renumber(sub_graph)

    tensor_names.propagate_op_name_to_tensor(graph)

    ir_path_suffix = "_tmp" if use_temporary_path else ""

    bin_file = os.path.join(output_dir, '{}{}.bin'.format(output_model_name, ir_path_suffix))
    serialize_constants(graph, bin_file)

    mean_offset = None
    mean_size = None
    if mean_data:
        mean_offset, mean_size = serialize_mean_image(bin_file, mean_data=mean_data)

    generate_ie_ir(graph=graph,
                   file_name=os.path.join(output_dir, '{}{}.xml'.format(output_model_name, ir_path_suffix)),
                   input_names=input_names,
                   mean_offset=mean_offset,
                   mean_size=mean_size,
                   meta_info=meta_info)
    tensor_names.output_tensor_names_map(graph, os.path.join(output_dir, '{}{}.mapping'.format(output_model_name, ir_path_suffix)))


def get_ir_version(argv: argparse.Namespace):
    """
    Determine IR version based on command line arguments and the default version.
    :param argv: the parsed command line arguments
    :return: the IR version
    """
    return 10
