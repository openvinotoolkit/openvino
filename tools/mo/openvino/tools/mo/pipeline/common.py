# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging as log
import os
from operator import itemgetter

import networkx as nx
import numpy as np

from openvino.tools.mo.back.RemoveUselessConvert import RemoveUselessConvert
from openvino.tools.mo.back.ResultRename import ResultRename
from openvino.tools.mo.back.ie_ir_ver_2.emitter import port_renumber, serialize_constants, generate_ie_ir, \
    serialize_mean_image
from openvino.tools.mo.back.op_versioning import OpVersioning
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.middle.passes import tensor_names, convert_data_type
from openvino.tools.mo.middle.passes.convert_data_type import data_type_str_to_np
from openvino.tools.mo.middle.passes.infer import type_infer
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.utils.error import Error


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
        if node.soft_get('version') != "opset11":
            # opset11 cannot be produced by legacy MO frontends, it can only be read by MO IR Reader
            if node.soft_get('type') in type_port:
                ports_to_update = type_port[node.soft_get('type')]
                for port_id, precision in ports_to_update.items():
                    if port_id in node.in_ports() and not node.in_port(port_id).disconnected():
                        log.debug('Converting value for the input port "{}" of op "{}" to "{}".'
                                  ''.format(port_id, node.soft_get('name', node.id), precision))
                        in_port = node.in_port(port_id)
                        np_type = data_type_str_to_np(precision)
                        in_node = node.in_port(port_id).get_source().node
                        in_type = in_node.out_port(0).get_data_type()

                        if in_node.type == 'Const':
                            if np.issubdtype(in_type, np.integer) and np.issubdtype(np_type, np.integer):
                                # do not convert Constant value if both source and destination types are of integer types
                                # otherwise, it affects compatibility of MO IR Engine and TF FE
                                # TF FE intents to use original model type for layers if it is possible
                                continue
                            convert_const_node_value_type(in_node, np_type)
                        else:
                            allowed_int_types = [np.int32, np.int64, np.uint32, np.uint64]
                            if in_type in allowed_int_types and np_type in allowed_int_types:
                                # do not convert if both source and destination types are within the set of
                                # int32/int64/uint32/uint64. It prevents from getting different IRs from the original
                                # cpp serializer and from the legacy serialized when restored with ir_reader_utils
                                continue
                            in_port.get_connection().insert_node(Cast(graph, {'dst_type': np_type}).create_node())


def set_default_tensor_names_for_parameters_results(graph: Graph):
    for node in graph.get_op_nodes():
        if node.soft_get('type') == 'Result' and node.is_in_port_connected(0):
            port = node.in_port(0).get_connection().get_source()
        elif node.soft_get('type') == 'Parameter' and node.is_out_port_connected(0):
            port = node.out_port(0)
        else:
            continue
        if node.has_and_set('keep_output_port'):
            continue

        tensors = port.get_tensor_names()
        if tensors is not None and isinstance(tensors, list) and len(tensors) > 0:
            continue
        new_tensor_name = port.get_default_tensor_name()
        op_name = port.node.soft_get('name')
        port.add_tensor_names([new_tensor_name, op_name])


def prepare_emit_ir(graph: Graph, data_type: str, output_dir: str, output_model_name: str,
                    mean_data: [list, None] = None, input_names: list = None, meta_info: dict = None,
                    use_temporary_path=False, convert_types=False, rename_results=True):
    if input_names is None:
        input_names = []
    if meta_info is None:
        meta_info = {}
    graph.strict_mode = False

    if convert_types:
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

    for_graph_and_each_sub_graph_recursively(graph, RemoveUselessConvert().find_and_replace_pattern)

    if rename_results:
        ResultRename().find_and_replace_pattern(graph)
    set_default_tensor_names_for_parameters_results(graph)

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
    tensor_names.output_tensor_names_map(graph, os.path.join(output_dir,
                                                             '{}{}.mapping'.format(output_model_name, ir_path_suffix)))


def get_ir_version(argv: argparse.Namespace):
    """
    Determine IR version based on command line arguments and the default version.
    :param argv: the parsed command line arguments
    :return: the IR version
    """
    return 11
