# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively

from .ops import OPERATIONS


class FunctionResultsAccumulator:
    """
    Run function on graph and store each call result in the list
    """
    def __init__(self, func):
        self._results = []
        self._func = func

    @property
    def results(self):
        return self._results

    def __call__(self, graph):
        result = self._func(graph)
        if not result is None:
            self._results.extend(result)


def find_node(graph: Graph, name):
    """ Get node by name
    :param graph: NetworkX model to search in
    :param name: node name in graph
    :return node instance in 'graph' or None if there's no node with such name
    """
    for op in graph.get_op_nodes():
        attrs = op.get_attrs()
        if 'name' in attrs and attrs['name'] == name:
            return op
    return None


# TODO: set recursively = True to enable subgraphs quantization
def get_node_by_name(graph: Graph, name: str, recursively: bool = False) -> Node:
    """ Returns node by name
    :param graph: NetworkX model to take node
    :param name: name of the node
    :param recursively: whether return all nodes from the graph
    and each subgraph or only from the external graph
    :return node from NetworkX model (of type Node or None if there's no such node)
    """
    if recursively:
        def get_node_by_fullname(graph: Graph, name: str) -> Node:
            nodes = graph.get_nodes_with_attributes(**dict(kind='op', fullname=name))
            return [Node(graph, nodes[0])] if nodes else None

        partial_get_node_by_fullname = partial(get_node_by_fullname, name=name)
        get_node_by_fullname_func = FunctionResultsAccumulator(partial_get_node_by_fullname)
        for_graph_and_each_sub_graph_recursively(graph, get_node_by_fullname_func)
        node = get_node_by_fullname_func.results
    else:
        node = graph.get_op_nodes(name=name)

    return node[0] if node else None


def remove_node_by_name(graph: Graph, node_name: str) -> (list, list):
    """ Removes node from graph
    :param graph: NetworkX model to update
    :param node_name: name of the node to remove
    :return tuple(list of parents names, list of children names)
    """
    node = get_node_by_name(graph, node_name)
    if node is None:
        raise Exception('There\'s no node with {} name'.format(node_name))
    graph.remove_node(node)


def connect_nodes(src_node, src_port, dst_node, dst_port):
    """ Connects two nodes with each other
    :param src_node: name of the input node
    :param src_port: index of the port for input node
    :param dst_node: name of the destination node
    :param dst_port: index of the port for destination node
     """
    src_node.out_port(src_port).connect(dst_node.in_port(dst_port))


def connect_nodes_by_name(graph: Graph, src_node_name, src_port, dst_node_name, dst_port):
    """ Connects two nodes with each other by their names
    :param graph: NetworkX model to update
    :param src_node_name: name of the input node
    :param src_port: index of the port for input node
    :param dst_node_name: name of the destination node
    :param dst_port: index of the port for destination node
     """
    src_node = get_node_by_name(graph, src_node_name, recursively=False)
    if src_node is None:
        raise Exception('There\'s no node with {} name'.format(src_node_name))
    dst_node = get_node_by_name(graph, dst_node_name, recursively=False)
    if dst_node is None:
        raise Exception('There\'s no node with {} name'.format(dst_node_name))

    connect_nodes(src_node, src_port, dst_node, dst_port)


# TODO: set recursively = True to enable subgraphs quantization
def get_all_operation_nodes(graph: Graph, recursively: bool = False):
    """ Returns sequence of all nodes in graph
    :param graph: NetworkX model to take nodes
    :param recursively: whether return all nodes from the graph
    and each subgraph or only from the external graph
    :return list of all nodes
    """
    if recursively:
        get_all_op_nodes_func = FunctionResultsAccumulator(lambda graph: graph.get_op_nodes())
        for_graph_and_each_sub_graph_recursively(graph, get_all_op_nodes_func)
        return get_all_op_nodes_func.results

    return graph.get_op_nodes()


# TODO: set recursively = True to enable subgraphs quantization
def get_nodes_by_type(graph: Graph, types: list, recursively: bool = False) -> list:
    """ Returns all nodes with type from types collection
     :param graph: NetworkX model to collect nodes
     :param types: list of required types
     :param recursively: whether return all nodes from the graph
     and each subgraph or only from the main graph
     :return list of nodes filtered by 'types' collection
      """
    def get_nodes_by_type_from_main_graph(graph, types):
        return [node for t in types for node in graph.get_op_nodes(type=t)]

    if recursively:
        partial_get_nodes_by_type = partial(get_nodes_by_type_from_main_graph, types=types)
        get_nodes_by_type_recursively = FunctionResultsAccumulator(partial_get_nodes_by_type)
        for_graph_and_each_sub_graph_recursively(graph, get_nodes_by_type_recursively)
        nodes = [node for node in get_nodes_by_type_recursively.results if node]
    else:
        nodes = get_nodes_by_type_from_main_graph(graph, types)
    return nodes


def add_fullname_for_nodes(graph: Graph):
    def set_fullname(graph, subgraphs=None):
        if subgraphs is None:
            subgraphs = []
        for node in graph:
            node = Node(graph, node)
            if node.has_valid('sub_graphs'):
                for sub_graph_name in node.sub_graphs:
                    subgraphs.append(node.name)
                    set_fullname(node[sub_graph_name], subgraphs)
                    subgraphs = subgraphs[:-1]
            node['fullname'] = '|'.join(subgraphs + [node.name])

    set_fullname(graph)


def create_node(graph: Graph, node_name, node_type, node_attrs):
    """ Create node in graph
    :param graph: NetworkX model
    :param node_name: node name
    :param node_type: node operation type
    :param node_attrs: node attributes
    :return new node
    """
    if node_type not in OPERATIONS:
        raise RuntimeError('{} operation is not supported'.format(node_type))

    node_attrs['name'] = node_name

    return OPERATIONS[node_type](graph, node_attrs).create_node()
