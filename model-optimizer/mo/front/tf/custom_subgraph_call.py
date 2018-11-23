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
from re import compile, match, findall

import copy
import networkx as nx
import numpy as np
import tensorflow as tf

from mo.front.common.find_unsupported_ops import find_unsupported_ops_subgraphs
from mo.front.common.layout import convert_shape, nhwc_to_nchw_permute, nchw_to_nhwc_permute
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import update_ie_fields
from mo.front.tf.extractors.utils import tf_tensor_shape
from mo.front.tf.partial_infer.tf import get_subgraph_output_tensors, tf_subgraph_infer, \
    add_node_def_to_subgraph, update_input_in_pbs
from mo.graph.graph import dump_graph_for_graphviz, unique_id, Node, get_outputs, get_inputs, merge_edge_props
from mo.utils.graph import nodes_matching_name_pattern, is_connected_component

nchw_to_nhwc_constant_name = 'IE_NCHW_TO_NHWC'
nhwc_to_nchw_constant_name = 'IE_NHWC_TO_NCHW'


def replace_subgraph_calls(graph: nx.MultiDiGraph, patterns_string: str):
    """
    The function replaces sub-graphs defined by the node names with single nodes that are executed using the TensorFlow.
    The patterns applied independently, so N patterns produce N TensorFlow call nodes.
    :param graph: networkX graph to operate on.
    :param patterns_string: comma separated list of node names patterns.
    """
    cycle_exist = False
    patterns = patterns_string.split(',')
    for pattern in patterns:
        log.info("Merging nodes using pattern '{}'".format(pattern))
        matched_nodes = nodes_matching_name_pattern(graph, pattern)
        if len(matched_nodes) != 0:
            merge_nodes(graph, matched_nodes)
            try:
                # the function 'find_cycle' raises exception if the cycle is not found
                nx.find_cycle(graph)
                cycle_exist = True
            except nx.exception.NetworkXNoCycle:
                cycle_exist = False
            if cycle_exist:
                log.warning("Graph contains a cycle after merging nodes using pattern '{}'".format(pattern))
    if cycle_exist:
        dump_graph_for_graphviz(graph)
        log.error('graph contains cycle after applying all merge node patterns')


def offload_unsupported_operations_to_tf(graph: nx.MultiDiGraph, unsupported_nodes: list):
    assert len(unsupported_nodes) != 0
    sub_graphs_list = find_unsupported_ops_subgraphs(graph, unsupported_nodes, tf_find_constant_inputs)
    for nodes_set in sub_graphs_list:
        merge_nodes(graph, nodes_set)


def offload_operations_to_tf(graph: nx.MultiDiGraph, op_names_patterns: str):
    """
    The function accepts the list of strings with operation names patterns. The patterns applied independently and nodes
    matching specific pattern are executed using the TF runtime.
    :param graph: networkX graph to operate on.
    :param op_names_patterns: string with regular expressions specifying operation names patterns.
    """
    patterns = op_names_patterns.split(',')
    for pattern in patterns:
        log.info("Running nodes with operation using pattern '{}'".format(pattern))
        compiled_pattern = compile(pattern)
        for node_name, attrs in list(graph.nodes(data=True)):
            if 'pb' in graph.node[node_name]:
                op = graph.node[node_name]['pb'].op
                if match(compiled_pattern, op):
                    log.debug("Node '{}' operation matches pattern '{}'".format(node_name, pattern))
                    merge_nodes(graph, [node_name])


def make_shape_4d(shape: np.array):
    """
    Create 4D tensor from 1D, 2D or 3D by adding new dimensions of size 1.
    :param shape: shape to extend.
    :return: 4D tensor.
    """
    new_shape = int64_array(shape)
    old_shape_len = len(shape)

    for x in range(4 - old_shape_len):  # TODO think about proper way to add additional dimensions considering layout
        if len(new_shape) <= 1:  # if the shape is 0D or 1D then we should add additional dimensions to batch dimension
            new_shape = np.insert(new_shape, 0, 1)
        #            new_shape = np.array([1, shape[0], 1, 1])
        else:
            new_shape = np.insert(new_shape, 1, 1)
    return new_shape


def add_reshape_before_op_node(graph: nx.MultiDiGraph, data_node_name: str, op_node_name: str, edge_attrs: dict):
    """
    Adds reshape operation which expands dimension of the specified data tensor to 4D.
    :param graph: graph to operate on.
    :param data_node_name: the name of the data node to be reshaped to 4D tensor.
    :param op_node_name: name of the TFCustomSubgraphCall node which produces the tensor.
    :param edge_attrs: edge attributes which should be preserved.
    :return: None
    """
    data_node = Node(graph, data_node_name)

    graph.remove_edge(data_node_name, op_node_name)

    assert data_node['shape'] is not None

    new_shape = make_shape_4d(data_node['shape'])

    # reshape shape data node
    reshape_shape_data_node_name = unique_id(graph, "Reshape_shape_")
    graph.add_node(reshape_shape_data_node_name, kind='data', precision="FP32", name=reshape_shape_data_node_name,
                   value=new_shape, shape=[1])

    # reshape operation node
    reshape_node_name = unique_id(graph, "Reshape_")
    graph.add_node(reshape_node_name, kind='op', precision="FP32", type='Reshape', name=reshape_node_name, op='Reshape',
                   data_type=data_node['data_type'])
    update_ie_fields(graph.node[reshape_node_name])

    # reshaped data node
    reshaped_value = None
    if data_node['value'] is not None:
        reshaped_value = np.reshape(data_node['value'], new_shape)
    reshaped_data_node_name = unique_id(graph, "reshaped_data_")
    graph.add_node(reshaped_data_node_name, kind='data', precision="FP32", name=reshaped_data_node_name,
                   shape=new_shape, value=reshaped_value, nchw_layout=True)

    graph.add_edges_from([
        (data_node_name, reshape_node_name, {'in': 0}),
        (reshape_shape_data_node_name, reshape_node_name, {'in': 1}),
        (reshape_node_name, reshaped_data_node_name, {'out': 0}),
        (reshaped_data_node_name, op_node_name, edge_attrs)
    ])


def add_reshape_after_data_node(graph: nx.MultiDiGraph, data_node_name: str):
    """
    Adds reshape operation which changes shape of the tensor produced by TFSubgraphCall from 4D to real dimension
    of the tensor. The data_node_name node contains real dimensions of the tensor but they will be changed in the
    add_reshapes_for_tf_subgraph_calls function to a 4D because IE TF call layer supports output in 4D only.
    :param graph: graph to operate on.
    :param data_node_name: name of the data node to be reshaped to correct dimensions.
    :return: None
    """
    data_node = Node(graph, data_node_name)

    # if the data node was previously marked as output then we need to mark as output new reshaped data node
    is_output = False
    if data_node.has_and_set('is_output'):
        is_output = data_node['is_output']
        data_node['is_output'] = False

    # save old consumers nodes with edge attributes
    old_consumer_nodes_with_attrs = list()
    for index, out_op in enumerate(data_node.out_nodes()):
        edge_attrs = graph.get_edge_data(data_node_name, out_op.name)[0]
        old_consumer_nodes_with_attrs.append((out_op.name, edge_attrs))

    # remove old consumers from the data node
    for out_op in list(data_node.out_nodes()):
        graph.remove_edge(data_node_name, out_op.name)

    # reshape operation node
    reshape_node_name = unique_id(graph, "Reshape_")
    graph.add_node(reshape_node_name, kind='op', precision="FP32", type='Reshape', name=reshape_node_name, op='Reshape',
                   data_type=data_node['data_type'])
    update_ie_fields(graph.node[reshape_node_name])

    # reshape shape data node
    reshape_shape_data_node_name = unique_id(graph, "Reshape_shape_")
    graph.add_node(reshape_shape_data_node_name, kind='data', precision="FP32", name=reshape_shape_data_node_name,
                   value=np.array(data_node['shape']), shape=[1])

    # reshaped data node
    reshaped_value = None
    if data_node['value'] is not None:
        reshaped_value = np.array(data_node['value'])
    reshaped_data_node_name = unique_id(graph, "reshaped_data_")
    graph.add_node(reshaped_data_node_name, kind='data', precision="FP32", name=reshaped_data_node_name,
                   shape=np.array(data_node['shape']), value=reshaped_value, is_output=is_output, nchw_layout=True)

    graph.add_edges_from([
        (data_node_name, reshape_node_name, {'in': 0}),
        (reshape_shape_data_node_name, reshape_node_name, {'in': 1}),
        (reshape_node_name, reshaped_data_node_name, {'out': 0}),
    ])

    for out_node_name, edge_attrs in old_consumer_nodes_with_attrs:
        graph.add_edges_from([
            (reshaped_data_node_name, out_node_name, edge_attrs)
        ])


def add_reshapes_for_tf_subgraph_calls(graph: nx.MultiDiGraph):
    """
    Input and output tensors of the TFCustomSubgraphCall must be 4D because IE layer accepts and produces only 4D
    tensors. This function adds reshape operations where it is necessary.
    :param graph: graph to operate on.
    :return: None.
    """
    for src_node_name, dst_node_name, edge_attrs in list(graph.edges(data=True)):
        src_node = Node(graph, src_node_name)
        dst_node = Node(graph, dst_node_name)
        if dst_node.kind == 'op' and dst_node.has_valid('type') and dst_node.type == 'TFCustomSubgraphCall' and \
                src_node.has_valid('shape') and len(src_node.shape) != 4:
            log.info("There is an data tensor of shape '{}' which goes into '{}' node".format(
                src_node.shape, dst_node.type))
            add_reshape_before_op_node(graph, src_node_name, dst_node_name, edge_attrs)

    for node_name in list(graph.nodes()):
        node = Node(graph, node_name)
        if node['kind'] == 'op' and node.has_and_set('type') and node.type == 'TFCustomSubgraphCall':
            for index, data_node in node.out_nodes().items():
                real_dims_count = len(data_node.shape)
                if real_dims_count != 4:
                    log.info("There is an data tensor of shape '{}' with real dims count '{}' which goes out of '{}' "
                             "node".format(data_node.shape, real_dims_count, node.name))
                    add_reshape_after_data_node(graph, data_node.id)

                    # need to update shape of the op so IE generates XML with 4D tensors
                    out_shape = make_shape_4d(data_node['shape'])

                    data_node['shape'] = out_shape


def internal_output_name_for_node(node_name: str, output_port: int):
    return node_name + ":" + str(output_port)


def add_node_pb_if_not_yet_added(node: Node, mega_node: Node):
    if node.has_valid('pb') and node.pb.name not in mega_node.pbs.keys():
        mega_node.pbs[node.pb.name] = node.pb


def find_input_port(node: Node, input_desc: list, search_node_name: str, search_node_port: int):
    if input_desc is None:
        return len(node.in_nodes())

    for in_port, tensor_desc in enumerate(input_desc):
        for node_pattern, node_port in tensor_desc:
            if findall(node_pattern, search_node_name) and node_port == search_node_port:
                return in_port
    raise Exception('Did not find input port of the node "{}" with port "{}"'.format(search_node_name,
                                                                                     search_node_port))


def find_output_port(node: Node, output_desc: list, search_node_name: str, search_node_port: int):
    if output_desc is None:
        return len(node.out_nodes())

    for out_port, (node_pattern, node_port) in enumerate(output_desc):
        if findall(node_pattern, search_node_name) and node_port == search_node_port:
            return out_port
    raise Exception('Did not find output port of the node "{}" with port "{}"'.format(search_node_name,
                                                                                      search_node_port))


def merge_nodes(graph: nx.MultiDiGraph, nodes_to_merge_names: list, inputs_desc: list = None,
                outputs_desc: list = None):
    """
    Merges nodes specified in the set 'nodes_to_merge_names' into one mega-node, creating new edges between mega-node
    and inputs/outputs nodes of the mega-node. The added edges contain name of input/output nodes which will be used for
    generation of placeholders and will be saved to the IR xml so IE plug-in know how to map input/output data for the
    layer. Also the function adds protobufs of the nodes of the sub-graph and 'Const' ops consumed by nodes in the
    sub-graph to the node's attribute 'pbs'.
    :param graph: the graph object to operate on.
    :param nodes_to_merge_names: list of nodes names that should be merged into a single node.
    :param inputs_desc: optional list describing input nodes order.
    :param outputs_desc: optional list describing output nodes order.
    """
    if not is_connected_component(graph, nodes_to_merge_names):
        log.warning("The following nodes do not form connected sub-graph: {}".format(nodes_to_merge_names))
        dump_graph_for_graphviz(graph, nodes_to_dump=nodes_to_merge_names)

    new_node_name = unique_id(graph, "TFSubgraphCall_")
    log.info("Create new node with name '{}' for nodes '{}'".format(new_node_name, ', '.join(nodes_to_merge_names)))
    graph.add_node(new_node_name)
    new_node_attrs = graph.node[new_node_name]

    new_node_attrs['name'] = new_node_name
    set_tf_custom_call_node_attrs(new_node_attrs)
    new_node = Node(graph, new_node_name)

    added_input_tensors_names = set()  # set of tensors that are were added as input to the sub-graph
    added_new_node_output_tensors = dict()  # key - tensor name, value - out port

    for node_name in nodes_to_merge_names:
        node = Node(graph, node_name)
        add_node_pb_if_not_yet_added(node, new_node)
        for in_node_name, edge_attrs in get_inputs(graph, node_name):
            in_node = Node(graph, in_node_name)

            # internal edges between nodes of the sub-graph
            if in_node_name in nodes_to_merge_names:
                add_node_pb_if_not_yet_added(in_node, new_node)
                continue

            # edge outside of sub-graph into sub-graph
            if in_node_name not in nodes_to_merge_names:
                # we cannot use the 'in_node_name' as a protobuf operation name here
                # because the 'in_node_name' could be a sub-graph matched before.
                input_tensor_name = node.pb.input[edge_attrs['in']]
                if input_tensor_name not in added_input_tensors_names:
                    graph.add_edge(in_node_name, new_node_name,
                                   **merge_edge_props(
                                       {'in': find_input_port(new_node, inputs_desc, node_name, edge_attrs['in']),
                                        'out': edge_attrs['out'],
                                        'internal_input_node_name': input_tensor_name,
                                        'original_dst_node_name': node_name,
                                        'original_dst_port': edge_attrs['in'],
                                        'in_attrs': ['in', 'internal_input_node_name', 'original_dst_node_name',
                                                     'original_dst_port', 'placeholder_name'],
                                        'out_attrs': ['out']},
                                       edge_attrs)
                                   )
                    log.debug("Creating edge from outside of sub-graph to inside sub-graph: {} -> {}".format(
                        in_node_name, new_node_name))
                    added_input_tensors_names.add(input_tensor_name)

        # edge from inside sub-graph to outside sub-graph
        for out_node_name, edge_attrs in get_outputs(graph, node_name):
            if out_node_name not in nodes_to_merge_names:
                log.debug("Creating edge from inside of sub-graph to outside sub-graph: {} -> {}".format(
                    new_node_name, out_node_name))
                out_name = internal_output_name_for_node(node_name, edge_attrs['out'])
                if out_name not in added_new_node_output_tensors.keys():
                    added_new_node_output_tensors[out_name] = find_output_port(new_node, outputs_desc, node_name,
                                                                               edge_attrs['out'])
                graph.add_edge(new_node_name, out_node_name,
                               **merge_edge_props(
                                   {'in': edge_attrs['in'],
                                    'out': added_new_node_output_tensors[out_name],
                                    'internal_output_node_name': out_name,
                                    'in_attrs': ['in', 'internal_input_node_name'],
                                    'out_attrs': ['out', 'internal_output_node_name']},
                                   edge_attrs)
                               )
        new_node['output_tensors_names'] = [val for val in
                                            {v: k for k, v in added_new_node_output_tensors.items()}.values()]

    # add nodes using the same order as in initial GraphDef so we can dump them to IR in "correct" order
    new_node['nodes_order'] = [node for node in graph.graph['initial_nodes_order'] if node in new_node['pbs'].keys()]

    for n in nodes_to_merge_names:
        if graph.has_node(n):  # check if not deleted by another (similar) pattern
            graph.remove_node(n)
    return Node(graph, new_node_name)


def set_tf_custom_call_node_attrs(node_attrs: dict):
    update_ie_fields(node_attrs)
    node_attrs['input_nodes_names'] = list()
    node_attrs['output_tensors_names'] = list()
    node_attrs['real_input_dims'] = list()
    node_attrs['pbs'] = dict()
    node_attrs['type'] = 'TFCustomSubgraphCall'
    node_attrs['op'] = 'TFCustomSubgraphCall'
    node_attrs['precision'] = 'FP32'  # TODO use real precision derived from the model
    node_attrs['infer'] = tf_subgraph_infer
    node_attrs['kind'] = 'op'


def prepare_tf_call_nodes(graph: nx.MultiDiGraph):
    """
    The function performs preparation of the TF call nodes. Details are provided in the description of called functions.
    :param graph: graph to operate on.
    :return: None
    """
    update_placeholders(graph)
    add_output_nodes_transposes(graph)
    add_reshapes_for_tf_subgraph_calls(graph)


def update_placeholders(graph: nx.MultiDiGraph):
    """
    Iterates over all nodes of the graph, find all TF sub-graph call operations and updates placeholders shapes and adds
    transpose operation if necessary.
    :param graph: graph to operate on
    :return: None
    """
    for node_name in graph.nodes():
        node = Node(graph, node_name)
        if node.kind == 'op' and node.has_valid('op') and node.op == 'TFCustomSubgraphCall':
            update_placeholder_shape_and_add_transpose(node)


def update_placeholder_shape_and_add_transpose(node: Node):
    """
    The function changes placeholders shapes from NHWC to NCHW format and add transpose operations if needed.
    :param node: node to operate on.
    :return: None
    """
    tf.reset_default_graph()

    inputs_replacements = list()

    # transpose permutation constant
    nchw_to_nhwc_constant = tf.constant(nchw_to_nhwc_permute, dtype=tf.int32, name=nchw_to_nhwc_constant_name)
    nhwc_to_nchw_constant = tf.constant(nhwc_to_nchw_permute, dtype=tf.int32, name=nhwc_to_nchw_constant_name)

    for placeholder_name in node['input_nodes_names']:
        # dummy node which we can refer to as input in the transpose for the output node
        # dummy node should be unique for each placeholder
        dummy_node = tf.constant(value=[[[[1]]]], dtype=tf.float32, name='random_dummy_name_' + placeholder_name)

        placeholder = node['pbs'][placeholder_name]
        cur_shape = tf_tensor_shape(placeholder.attr['shape'].shape)
        if len(cur_shape) == 4:  # TODO think about better check that transpose is required
            nchw_shape = convert_shape(cur_shape, nhwc_to_nchw_permute)
            for ind in range(len(cur_shape)):
                placeholder.attr['shape'].shape.dim[ind].size = nchw_shape[ind]
            transpose_name = placeholder.name + '_transpose'
            transpose = tf.transpose(dummy_node, nchw_to_nhwc_constant, transpose_name)  # NCHW -> NHWC

            # add transpose operations to GraphDef after placeholders
            add_node_def_to_subgraph(node, transpose.op.node_def, transpose_name, len(node['input_nodes_names']))
            inputs_replacements.append((placeholder.name, transpose_name))
            inputs_replacements.append((dummy_node.name, placeholder.name))
            node['real_input_dims'].append(nchw_shape)
        else:
            node['real_input_dims'].append(cur_shape)
    add_node_def_to_subgraph(node, nchw_to_nhwc_constant.op.node_def)
    add_node_def_to_subgraph(node, nhwc_to_nchw_constant.op.node_def)

    # update initial input names to a transposed ones
    for old_input_tensor_name, new_name in inputs_replacements:
        update_input_in_pbs(node, old_input_tensor_name, new_name)


def add_output_nodes_transposes(graph: nx.MultiDiGraph):
    """
    Iterates over all nodes of the graph, find all TF sub-graph call operations and adds Transpose operations to the
    output nodes if they are 4D to covert output from NHWC to NCHW.
    :param graph: graph to operate on
    :return: None
    """
    for node_name in graph.nodes():
        node = Node(graph, node_name)
        if node.kind == 'op' and node.has_valid('op') and node.op == 'TFCustomSubgraphCall':
            add_sub_graph_call_output_tensors_transposes(node)


def add_sub_graph_call_output_tensors_transposes(node: Node):
    """
    Adds transpose operations to the output nodes if they are 4D to change layout from NCHW to NHWC.
    :param node: the node to add transposes to the output nodes to.
    :return: None
    """
    _, output_tensors = get_subgraph_output_tensors(node)

    # transpose permutation constant
    nhwc_to_nchw_constant = tf.constant(nhwc_to_nchw_permute, dtype=tf.int32, name=nhwc_to_nchw_constant_name)

    # dummy node which we can refer to as input in the transpose for the output node
    dummy_node = tf.constant(value=[[[[1]]]], dtype=tf.float32, name='random_dummy_name')

    new_out_tensor_names = list()
    for out_tensor_name in node['output_tensors_names']:
        out_name, out_port = out_tensor_name.split(':')
        if len(output_tensors[int(out_port)].shape) == 4:  # TODO think about better check whether transpose is required
            out_transpose_name = out_name + '_port_' + out_port + '_transpose'
            transpose = tf.transpose(dummy_node, nhwc_to_nchw_constant, name=out_transpose_name)

            # starting from TF 1.8 it is not possible to modify the "node_def" of the "tf.op", so we create a copy,
            # update it and use further
            new_input_names = transpose.op.node_def.input[:]
            new_input_names[0] = out_tensor_name
            new_node_def = copy.deepcopy(transpose.op.node_def)
            new_node_def.input[:] = new_input_names
            add_node_def_to_subgraph(node, new_node_def, position=len(node['nodes_order']))
            new_out_tensor_names.append(out_transpose_name)
        else:
            new_out_tensor_names.append(out_tensor_name)

    # update output tensor names with transposes operations
    node['output_tensors_names'] = new_out_tensor_names


def tf_find_constant_inputs(node: Node):
    """
    The function finds constant inputs of the node and nodes with Identity operation.
    :param node: node to add constants inputs.
    :return: set of added nodes (Node).
    """
    added_nodes = set()
    for in_node in node.in_nodes().values():
        if in_node.has_valid('pb'):
            if in_node['pb'].op == 'Const':
                added_nodes.add(in_node)
            if in_node['pb'].op == 'Identity':
                added_nodes.update(tf_find_constant_inputs(in_node))
    return added_nodes
