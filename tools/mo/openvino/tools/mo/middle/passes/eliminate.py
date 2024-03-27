# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import re
from collections import deque

import networkx as nx
import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import is_fully_defined, compatible_shapes
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import deprecated_api


# TODO: dep warning
def get_nodes_with_attributes(graph, **attrs: dict):
    node_attrs = graph.nodes(data=True)
    return [n for n, d in node_attrs if all(a in d.items() for a in attrs.items())]


def reverse_dfs(graph, node_name: str, update_func: callable, visited: set = None):
    d = deque()

    if visited is None:
        visited = set()
    visited.add(node_name)
    d.appendleft(node_name)
    while len(d) != 0:
        cur_node = d.popleft()
        update_func(graph, cur_node)
        for in_node_name, _ in graph.in_edges(cur_node):
            if in_node_name not in visited:
                visited.add(in_node_name)
                d.append(in_node_name)


def mark_input_nodes(graph, node_name: str, key: str, value):
    for input, _ in graph.in_edges(node_name):
        graph.node[input][key] = value


def mark_output_nodes(graph, node_name: str, key: str, value):
    for output, _ in graph.out_edges(node_name):
        graph.node[output][key] = value


def mark_output_reachable_nodes(graph):
    """
    Mark nodes whether they are outputs reachable or not. The node is considered output reachable if it is connected to
    one of the nodes that has attribute op=Result.
    """
    nx.set_node_attributes(G=graph, name='is_output_reachable', values=False)
    outputs = graph.get_nodes_with_attributes(op='Result')
    log.debug('The following nodes are seeded as output reachable:\n{}'.format('\n'.join(sorted(map(str, outputs)))))
    nx.set_node_attributes(G=graph, name='is_output_reachable', values={n: True for n in outputs})
    visited = set()
    for output_name in outputs:
        reverse_dfs(graph, output_name,
                    lambda graph, node_name: mark_input_nodes(graph, node_name, 'is_output_reachable', True), visited)


def mark_undead_nodes(graph, undead_types: list):
    """
    Mark output nodes and nodes of the specific type as undead, meaning that they should survive the dead nodes
    elimination phase. Then mark all children nodes of the undead nodes (except children of inputs) as undead.
    :param graph: graph to operate on.
    :param undead_types: list of node types that should be marked as undead.
    :return: updated graph where each has attribute 'is_undead'.
    """
    from openvino.tools.mo.utils.graph import bfs_search

    nx.set_node_attributes(G=graph, name='is_undead', values=False)

    undead_types_with_result = undead_types + ['Result']
    undead_nodes = []
    for node in graph.get_op_nodes():
        node_type = node.soft_get('type', node.soft_get('op'))
        if node_type in undead_types_with_result:
            undead_nodes.append(node.id)

    nx.set_node_attributes(G=graph, name='is_undead', values={n: True for n in undead_nodes})
    # propagate 'undead' attribute to children nodes of undead nodes if the node produces constant value
    for node_name in bfs_search(graph, undead_nodes):
        if graph.node[node_name]['is_undead']:
            for _, dst_node_name in graph.out_edges(node_name):
                node_attrs = graph.node[dst_node_name]
                if 'kind' in node_attrs and (
                        node_attrs['kind'] == 'data' and node_attrs['value'] is not None or node_attrs['kind'] == 'op'):
                    graph.node[dst_node_name]['is_undead'] = True

    # mark input nodes as undead
    inputs = graph.get_nodes_with_attributes(is_input=True)
    nx.set_node_attributes(G=graph, name='is_undead', values={n: True for n in inputs})


def mark_const_producer_nodes(graph):
    """
    Mark nodes that produce constant values.
    :param graph: graph to operate on.
    :return: .
    """
    nx.set_node_attributes(G=graph, name='is_const_producer', values=True)

    for node in graph.pseudo_topological_sort():
        for input, output, attrs in graph.in_edges(node.id, data=True):
            if 'control_flow_edge' in attrs and attrs['control_flow_edge']:
                graph.node[input]['is_const_producer'] = False
                graph.node[output]['is_const_producer'] = False

        if not node.has('value') or node.value is None or not is_fully_defined(node.value):
            for input, _ in graph.in_edges(node.id):
                graph.node[input]['is_const_producer'] = False


def eliminate_dead_nodes(graph):
    from openvino.tools.mo.graph.graph import Node
    nodes_to_remove = set()
    for node_name, node_attrs in graph.nodes(data=True):
        # The Const operation node may have set an attribute 'nchw_layout' attribute to prevent shape permutation.
        # During graph clean-up the operation node is removed and the attribute is lost.
        # This results in permutation of the Const shape in the IR and wrong inference results.
        # Here we explicitly save the 'nchw_layout' attribute in the data node to prevent permutation."
        if node_attrs.get('type', None) == 'Const':
            if node_attrs.get('nchw_layout', False):
                Node(graph, node_name).out_node()['nchw_layout'] = True
            if np.all(node_attrs.get('force_shape', False)):
                Node(graph, node_name).out_node()['force_shape'] = node_attrs['force_shape']
            if node_attrs.get('force_type', False):
                Node(graph, node_name).out_node()['force_type'] = node_attrs['force_type']

        if not node_attrs['is_output_reachable'] or \
                (node_attrs['is_const_producer'] and (not node_attrs['is_undead'] or
                                                      node_attrs.get('force_dead_node', False))):
            nodes_to_remove.add(node_name)
    log.debug('Removing the following dead nodes: {}'.format('\n'.join(sorted(map(str, nodes_to_remove)))))
    graph.remove_nodes_from(nodes_to_remove)


def add_constant_operations(graph):
    data_nodes = graph.get_data_nodes(has_value=True)
    for node in data_nodes:
        # If data node has no producers we create Const operation
        if len(node.in_nodes()) == 0 and len(node.out_nodes()) != 0:
            # It's necessary to import here due to cycle dependencies
            from openvino.tools.mo.ops.const import Const
            from openvino.tools.mo.utils.runtime_info import RTInfo
            name = node.soft_get('name', node.id)
            new_name = re.sub(r'\/Output_\d+\/Data_(.?)+', '', name)
            const_node = Const(graph, dict(value=node.value, name=new_name,
                                           force_shape=node.soft_get('force_shape', None),
                                           override_output_shape=node.has_valid('force_shape'),
                                           force_type=node.soft_get('force_type', None),
                                           correct_data_type=node.soft_get('correct_data_type', False),
                                           rt_info=node.soft_get('rt_info', RTInfo()),
                                           )).create_node()
            graph.add_edges_from([(const_node.id, node.id, {'out': 0})])


def shape_inference(graph):
    for node in graph.pseudo_topological_sort():
        if node.has_and_set('need_shape_inference'):
            old_out_shapes = [port.data.get_shape() for port in node.out_ports().values() if not port.disconnected()]
            node.infer(node)
            new_out_shapes = [port.data.get_shape() for port in node.out_ports().values() if not port.disconnected()]
            if not node.has_and_set('override_output_shape'):
                for shape1, shape2 in zip(old_out_shapes, new_out_shapes):
                    # do not use strict shapes comparison because after applying transformation the output shape may be
                    # specialized and some dynamic dimension become static
                    if shape1 is not None and not compatible_shapes(shape1, shape2):
                        raise Error("After partial shape inference were found shape collision for node {} (old shape: "
                                    "{}, new shape: {})".format(node.name, shape1, shape2))
            else:
                del node['override_output_shape']
            node.need_shape_inference = False


@deprecated_api('Graph', 'clean_up')
def graph_clean_up(graph, undead_node_types: list = None):
    graph.clean_up(undead_node_types)


@deprecated_api('Graph', 'clean_up')
def graph_clean_up_tf(graph):
    graph.clean_up()


@deprecated_api('Graph', 'clean_up')
def graph_clean_up_onnx(graph):
    graph.clean_up()


# TODO: unit tests
def merge_data_nodes(graph, survived, removed):
    if survived.has_and_set('op') and survived.op == 'Result':
        graph.node[removed.id].update({'op': 'Result'})

    for u, v, d in list(graph.in_edges(removed.id, data=True)):
        graph.add_edges_from([(u, survived.id, d)])
        graph.remove_edge(u, v)

    for u, v, d in list(graph.out_edges(removed.id, data=True)):
        graph.add_edges_from([(survived.id, v, d)])
        graph.remove_edge(u, v)

    for attr in graph.node[removed.id]:
        if not attr in ['name']:
            # We need to save debug info from removed data node
            if attr == 'fw_tensor_debug_info':
                if not survived.has_valid(attr):
                    survived[attr] = []
                for fw_tensor_debug_info in removed[attr]:
                    survived[attr].append(fw_tensor_debug_info)
            else:
                survived[attr] = removed[attr]


# TODO: unit tests
def remove_op_node_with_data_node(graph, node_to_remove, input_data_node=None):
    from openvino.tools.mo.graph.graph import Node
    assert node_to_remove.kind == 'op'
    if input_data_node is None:
        input_data_node = node_to_remove.in_node()
    output_node = [v for _, v in graph.out_edges(node_to_remove.id)]
    assert len(output_node) == 1, "Cannot remove node producing two or more output tensors"
    output_node = Node(graph, output_node[0])
    assert output_node.kind == 'data', "The function must be used after partial infer"

    graph.remove_edge(input_data_node.id, node_to_remove.id)
    graph.remove_edge(node_to_remove.id, output_node.id)

    merge_data_nodes(graph, output_node, input_data_node)

    # we just have saved all output edges from 'input' by reconnecting them to 'output', now we can delete 'input'
    log.debug('Removing op node: {}'.format(node_to_remove.id))
    graph.remove_nodes_from([node_to_remove.id, input_data_node.id])


def remove_op_nodes(graph, attrs: dict):
    for node in graph.get_op_nodes(**attrs):
        remove_op_node_with_data_node(graph, node)


def remove_edges_for_nodes(graph, node_attrs: dict, edge_attrs: dict):
    from openvino.tools.mo.graph.graph import Node
    for node in graph.nodes():
        node = Node(graph, node)
        if all([node.has(attr) and node[attr] == node_attrs[attr] for attr in node_attrs]):
            nodes_edges = node.in_nodes_edges()
            for port in nodes_edges:
                src_node, edge = nodes_edges[port]
                if all([attr in edge and edge[attr] == edge_attrs[attr] for attr in edge_attrs]):
                    graph.remove_edge(src_node.id, node.id)
