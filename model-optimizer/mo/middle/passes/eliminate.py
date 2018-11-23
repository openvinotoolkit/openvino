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
from collections import deque

import networkx as nx

from mo.graph.graph import Node, create_edge
from mo.middle.pattern_match import apply_pattern
from mo.utils.graph import bfs_search, pseudo_topological_sort


def get_nodes_with_attributes(graph: nx.MultiDiGraph, **attrs: dict):
    node_attrs = graph.nodes(data=True)
    return [n for n, d in node_attrs if all(a in d.items() for a in attrs.items())]


def reverse_dfs(graph: nx.MultiDiGraph, node_name: str, update_func: callable, visited: set = None):
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


def mark_input_nodes(graph: nx.MultiDiGraph, node_name: str, key: str, value):
    for input, _ in graph.in_edges(node_name):
        graph.node[input][key] = value


def mark_output_nodes(graph: nx.MultiDiGraph, node_name: str, key: str, value):
    for output, _ in graph.out_edges(node_name):
        graph.node[output][key] = value


def mark_output_reachable_nodes(graph: nx.MultiDiGraph):
    """
    Mark nodes whether they are outputs reachable or not. The node is considered output reachable if it is connected to
    one of the nodes that has attribute is_output=True.
    """
    nx.set_node_attributes(graph, name='is_output_reachable', values=False)
    outputs = get_nodes_with_attributes(graph, is_output=True)
    log.debug('The following nodes are seeded as output reachable:\n{}'.format('\n'.join(sorted(map(str, outputs)))))
    nx.set_node_attributes(graph, name='is_output_reachable', values={n: True for n in outputs})
    for output_name in outputs:
        reverse_dfs(graph, output_name,
                    lambda graph, node_name: mark_input_nodes(graph, node_name, 'is_output_reachable', True), set())


def mark_undead_nodes(graph: nx.MultiDiGraph, undead_types: list):
    """
    Mark output nodes and nodes of the specific type as undead, meaning that they should survive the dead nodes
    elimination phase. Then mark all children nodes of the undead nodes (except children of inputs) as undead.
    :param graph: graph to operate on.
    :param undead_types: list of node types that should be marked as undead.
    :return: updated graph where each has attribute 'is_undead'.
    """
    nx.set_node_attributes(graph, name='is_undead', values=False)

    # mark output nodes as undead
    outputs = get_nodes_with_attributes(graph, is_output=True)
    nx.set_node_attributes(graph, name='is_undead', values={n: True for n in outputs})

    # mark specifically defined with node type set of nodes
    for type in undead_types:
        node_of_specific_type = get_nodes_with_attributes(graph, type=type)
        nx.set_node_attributes(graph, name='is_undead', values={n: True for n in node_of_specific_type})

    undead_nodes = get_nodes_with_attributes(graph, is_undead=True)
    # propagate 'undead' attribute to children nodes of undead nodes if the node produces constant value
    for node_name in bfs_search(graph, undead_nodes):
        if graph.node[node_name]['is_undead']:
            for _, dst_node_name in graph.out_edges(node_name):
                node_attrs = graph.node[dst_node_name]
                if 'kind' in node_attrs and node_attrs['kind'] == 'data' and node_attrs['value'] is not None:
                    graph.node[dst_node_name]['is_undead'] = True

    # mark input nodes as undead
    inputs = get_nodes_with_attributes(graph, is_input=True)
    nx.set_node_attributes(graph, name='is_undead', values={n: True for n in inputs})


def mark_const_producer_nodes(graph: nx.MultiDiGraph):
    """
    Mark nodes that produce constant values.
    :param graph: graph to operate on.
    :return: .
    """
    nx.set_node_attributes(graph, name='is_const_producer', values=True)

    for n in pseudo_topological_sort(graph):
        node = Node(graph, n)
        for input, output, attrs in graph.in_edges(n, data=True):
            if 'control_flow_edge' in attrs and attrs['control_flow_edge']:
                graph.node[input]['is_const_producer'] = False
                graph.node[output]['is_const_producer'] = False

        if not node.has('value') or node.value is None:
            for input, _ in graph.in_edges(n):
                graph.node[input]['is_const_producer'] = False


def eliminate_dead_nodes(graph: nx.MultiDiGraph):
    nodes_to_remove = set()
    for node_name, node_attrs in graph.nodes(data=True):
        if not node_attrs['is_output_reachable'] or (node_attrs['is_const_producer'] and not node_attrs['is_undead']):
            nodes_to_remove.add(node_name)
    log.debug('Removing the following dead nodes: {}'.format('\n'.join(sorted(map(str, nodes_to_remove)))))
    graph.remove_nodes_from(nodes_to_remove)


def graph_clean_up(graph: nx.MultiDiGraph, undead_node_types: list = []):
    mark_output_reachable_nodes(graph)
    mark_undead_nodes(graph, undead_node_types)
    mark_const_producer_nodes(graph)
    eliminate_dead_nodes(graph)


def graph_clean_up_tf(graph: nx.MultiDiGraph):
    graph_clean_up(graph, ['TFCustomSubgraphCall'])


def remove_identity_action(graph: nx.MultiDiGraph, matches: dict):
    remove_op_node(graph, matches['identity'])


# TODO: unit tests
def merge_data_nodes(graph: nx.MultiDiGraph, survived: Node, removed: Node):
    if survived.has_and_set('is_output'):
        graph.node[removed.id].update({'is_output': True})

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
def remove_op_node(graph: nx.MultiDiGraph, identity: Node):
    input = identity.in_node()
    output = [v for _, v in graph.out_edges(identity.id)]
    assert len(output) == 1
    output = Node(graph, output[0])

    graph.remove_edge(input.id, identity.id)
    graph.remove_edge(identity.id, output.id)

    merge_data_nodes(graph, output, input)

    # we just have saved all output edges from 'input' by reconnecting them to 'output', now we can delete 'input'
    log.debug('Removing op node: {}'.format(identity.id))
    graph.remove_node(identity.id)
    graph.remove_node(input.id)


def remove_op_nodes(graph: nx.MultiDiGraph, attrs: dict):
    op_attrs = {'kind': 'op'}
    op_attrs.update(attrs)
    apply_pattern(
        graph,
        nodes=[('identity', op_attrs)],
        edges=[],
        action=remove_identity_action
    )


def remove_edges_for_nodes(graph: nx.MultiDiGraph, node_attrs: dict, edge_attrs: dict):
    for node in graph.nodes():
        node = Node(graph, node)
        if all([node.has(attr) and node[attr] == node_attrs[attr] for attr in node_attrs]):
            nodes_edges = node.in_nodes_edges()
            for port in nodes_edges:
                src_node, edge = nodes_edges[port]
                if all([attr in edge and edge[attr] == edge_attrs[attr] for attr in edge_attrs]):
                    graph.remove_edge(src_node.id, node.id)


def remove_useless_split_action(graph: nx.MultiDiGraph, matches: dict):
    split_node = matches['split']
    input = split_node.in_node(1)
    output = split_node.out_node()
    graph.remove_edge(input.id, split_node.id)

    for u, v, d in list(graph.out_edges(output.id, data=True)):
        graph.add_edges_from([(input.id, v, d)])
        graph.remove_edge(u, v)


def remove_useless_split(graph: nx.MultiDiGraph):
    apply_pattern(
        graph,
        nodes=[('split', {'kind': 'op', 'op': 'Split', 'num_split': 1})],
        edges=[],
        action=remove_useless_split_action
    )


def remove_node_from_graph(graph: nx.MultiDiGraph, previous_node: Node, removing_node: Node):
    if len(removing_node.out_nodes()) > 0:
        last_node_out = removing_node.out_node(0)
        edge_data = graph.get_edge_data(removing_node.id, last_node_out.id)
        out_port = edge_data[0]['out']
        in_port = edge_data[0]['in']
        graph.remove_edge(previous_node.id, removing_node.id)
        graph.remove_edge(removing_node.id, last_node_out.id)
        create_edge(previous_node, last_node_out, out_port=out_port, in_port=in_port)
        graph.remove_node(removing_node.id)
