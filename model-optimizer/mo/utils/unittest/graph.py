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

from collections import deque
from copy import deepcopy
from numbers import Number

import networkx as nx
import numpy as np

from mo.graph.graph import Node, Graph
from mo.middle.pattern_match import all_edges_in_nodes
from mo.utils.error import Error


def not_all_new(old_elements: list, new_elements: list):
    """
    This function check whether at least one element from new_elements are in old_elements.
    """
    return any([element in old_elements for element in new_elements])


def check_and_update_ports(node, edges_data: list, in_port: bool = True):
    key = 'in' if in_port else 'out'
    key_in_edges = [key in edge_data for edge_data in edges_data]
    if all(key_in_edges):
        ports = [edge_data[key] for edge_data in edges_data]
        if len(ports) != len(set(ports)):
            raise Error("Please, provide unique {} ports for nodes".format(key))
    elif not any(key_in_edges):
        if node.has_valid('kind') and node.kind == 'data':
            return
        for i, edge_data in enumerate(edges_data):
            edge_data[key] = i
    else:
        raise Error("Please, provide all {} ports for nodes".format(key))


def build_graph_with_attrs(nodes_with_attrs: list, edges_with_attrs: list, new_nodes_with_attrs: list = [],
                           new_edges_with_attrs: list = [], update_edge_attrs: dict = None,
                           update_nodes_attributes: dict = None, nodes_with_edges_only: bool = False,
                           add_nodes_from_edges: bool = False):
    """
    Build the Graph with specific nodes and edges. Also update of edge and node parameters is supported.
    :param nodes_with_attrs: list of tuples ('node_name', {node_attrs})
    :param edges_with_attrs: list of tuples like (start node, end node, (optional) {attrs of the edge}).
    :param new_nodes_with_attrs: analogically nodes_with_attrs
    :param new_edges_with_attrs: analogically new_edges
    :param update_edge_attrs: optional dictionary like {('from_node', 'to_node', key): {edge_attrs}}.
    :param update_nodes_attributes: optional dictionary which specifies nodes names and their attributes to be updated. The
    key is a node name to update attribute and the value is a dictionary with attribute name and its value.
    :param nodes_with_edges_only: add nodes which has at least one incoming or outcoming edge.
    :param add_nodes_from_edges: whether nodes that is not listed in all_nodes but are in all_edges is allowed.
    :return: generated graph.
    """
    if not_all_new([node[0] for node in nodes_with_attrs], [node[0] for node in new_nodes_with_attrs]):
        raise Error('Some nodes from new_nodes_with_attrs are already in nodes.'
                    ' Please, add to new_nodes_with_attrs only NEW nodes.')

    if not_all_new([(edge[0], edge[1]) for edge in edges_with_attrs],
                   [(edge[0], edge[1]) for edge in new_edges_with_attrs]):
        raise Error('Some edges from new_edges_with_attrs are already in edges.'
                    ' Please, add to new_edges_with_attrs only NEW edges.')

    # Check that all nodes from list of edges are in nodes
    all_nodes = nodes_with_attrs + new_nodes_with_attrs
    all_edges = edges_with_attrs + new_edges_with_attrs
    all_nodes_names = [node[0] for node in all_nodes]
    if not add_nodes_from_edges and not all_edges_in_nodes(nodes=all_nodes_names, edges=all_edges):
        raise Error("Some nodes from list of edges is not in nodes. Please, add all necessary nodes.")

    graph = Graph()

    # Create dict for nodes with attrs
    nodes_attrs = {}
    for node_name, attrs in all_nodes:
        nodes_attrs[node_name] = attrs
        if 'name' not in attrs:
            attrs['name'] = node_name

    if nodes_with_edges_only:
        # filter nodes to keep only ones with edges connected
        filtered_nodes = {}
        for edge in all_edges:
            node_1, node_2 = edge[0], edge[1]
            filtered_nodes[node_1] = nodes_attrs[node_1]
            filtered_nodes[node_2] = nodes_attrs[node_2]
        nodes_attrs = filtered_nodes

    # Create all nodes
    for node, attrs in nodes_attrs.items():
        graph.add_node(node, **deepcopy(attrs))

    # Connect nodes with edges (also unpack edge params)
    for edge in all_edges:
        node_1, node_2 = edge[0], edge[1]
        edge_attrs = edge[2] if len(edge) == 3 else {}
        graph.add_edge(node_1, node_2, **edge_attrs)

    # Update attributes of edges
    if update_edge_attrs:
        # it will work in 2.x networkx only
        for edge, attr in update_edge_attrs.items():
            for k, v in attr.items():
                nx.set_edge_attributes(G=graph, name=k, values={edge: v})

    # Update attributes of nodes
    if update_nodes_attributes is not None:
        for node_name, new_attrs in update_nodes_attributes:
            assert (node_name in graph.nodes())
            for attr, value in new_attrs.items():
                graph.node[node_name][attr] = value

    for node_id in graph.nodes():
        node = Node(graph, node_id)
        check_and_update_ports(node, [graph.get_edge_data(edge[0], node_id)[0] for edge in graph.in_edges(node_id)], True)
        check_and_update_ports(node, [graph.get_edge_data(node_id, edge[1])[0] for edge in graph.out_edges(node_id)], False)

    for node in graph.get_op_nodes():
        # Add in_ports attribute
        in_edges = node.in_edges()
        for i in range(len(in_edges)):
            node.add_input_port(idx=i)

        # Add out_ports attribute
        out_edges = node.out_edges()
        for i in range(len(out_edges)):
            node.add_output_port(idx=i)
    return graph


def build_graph(nodes_attrs: dict, edges: list, update_attributes: dict = None, nodes_with_edges_only: bool = False):
    """
    Build the Graph with specific nodes and edges.
    :param nodes_attrs: dictionary where key is the node name and the value is the dictionary with node attributes.
    :param edges: list of pairs with start and end node names of the edge.
    :param update_attributes: optional dictionary which specifies nodes names and their attributes to be updated. The
    key is a node name to update attribute and the value is a dictionary with attribute name and its value.
    :param nodes_with_edges_only: add nodes which has at least one incoming or outcoming edge.
    :return: generated graph.
    """
    graph = Graph()

    for node_name, attrs in nodes_attrs.items():
        if 'name' not in attrs:
            attrs['name'] = node_name

    if nodes_with_edges_only:
        # filter nodes to keep only ones with edges connected
        filtered_nodes = {}
        for item in edges:
            if len(item) == 2:  # TODO: is there any better way in python to do that?
                node1, node2 = item
            else:
                node1, node2, _ = item
            filtered_nodes[node1] = nodes_attrs[node1]
            filtered_nodes[node2] = nodes_attrs[node2]
        nodes_attrs = filtered_nodes

    # create all nodes first
    for node, attrs in nodes_attrs.items():
        assert node not in graph.nodes()
        graph.add_node(node, **deepcopy(attrs))

    # connect nodes with edges
    for item in edges:
        if len(item) == 2:  # TODO: is there any better way in python to do that?
            node_1, node_2 = item
            edge_attrs = {}
        else:
            node_1, node_2, edge_attrs = item

        common_attrs = {'in': len(graph.in_edges(node_2)),
                        'out': len(graph.out_edges(node_1)),
                        'name': nodes_attrs[node_1]['name']}
        common_attrs.update(edge_attrs)
        graph.add_edge(node_1, node_2, **common_attrs)

    if update_attributes is not None:
        for node_name, new_attrs in update_attributes.items():
            assert (node_name in graph.nodes()), 'Node with name "{}" is not in the graph'.format(node_name)
            for attr, value in new_attrs.items():
                graph.node[node_name][attr] = value

    for node in graph.get_op_nodes():
        # Add in_ports attribute
        in_edges = node.in_edges()
        for i in range(len(in_edges)):
            node.add_input_port(idx=i)

        # Add out_ports attribute
        out_edges = node.out_edges()
        for i in range(len(out_edges)):
            node.add_output_port(idx=i)

    return graph


def build_graph_with_edge_attrs(nodes_attrs: dict, edges: list, update_attributes: dict = None):
    """
    Build the Graph with specific nodes and edges.
    :param nodes_attrs: dictionary where key is the node name and the value is the dictionary with node attributes.
    :param edges: list of pairs with start and end node names of the edge.
    :param update_attributes: optional dictionary which specifies nodes names and their attributes to be updated. The
    key is a node name to update attribute and the value is a dictionary with attribute name and its value.
    :return: generated graph.
    """
    graph = Graph()
    for node_1, node_2, attr in edges:
        if node_1 not in graph.nodes():
            graph.add_node(node_1, **deepcopy(nodes_attrs[node_1]))
        if node_2 not in graph.nodes():
            graph.add_node(node_2, **deepcopy(nodes_attrs[node_2]))
        graph.add_edge(node_1, node_2, **attr)
    if update_attributes is not None:
        for node_name, new_attrs in update_attributes.items():
            assert (node_name in graph.nodes())
            for attr, value in new_attrs.items():
                graph.node[node_name][attr] = value
    return graph


def compare_graphs(graph: Graph, graph_ref: Graph, last_node: str, last_node_ref=None, check_op_attrs=False):
    if last_node_ref is None:
        last_node_ref = last_node

    q = deque([last_node])
    q_ref = deque([last_node_ref])

    checked_nodes = []
    checked_nodes_ref = []

    while len(q_ref) != 0:
        if len(q) == 0:
            return False, 'Graphs have different number of nodes'
        node = Node(graph, q.popleft())
        node_ref = Node(graph_ref, q_ref.popleft())

        checked_nodes.append(node.id)
        checked_nodes_ref.append(node_ref.id)

        # Check that nodes has same amount of output nodes
        if len(node_ref.out_nodes()) != len(node.out_nodes()):
            return False, 'Current node "{}" and reference node "{}" have different amount of output nodes: {} vs {}'.\
                format(node.id, node_ref.id, len(node_ref.out_nodes()), len(node.out_nodes()))

        # Check that nodes has same amount of input nodes
        if len(node_ref.in_nodes()) != len(node.in_nodes()):
            return False, 'Current node "{}" and reference node "{}" have different amount of input nodes: {} vs {}'.\
                format(node.id, node_ref.id, len(node_ref.in_nodes()), len(node.in_nodes()))

        # Check that nodes has same 'kind'
        if node_ref.kind != node.kind:
            return False, 'Current node "{}" and reference node "{}" have different kind parameter'.\
                format(node.id, node_ref.id)

        # Check can_be_fused attr
        if node_ref.has_valid('can_be_fused'):
            if node_ref.soft_get('can_be_fused') != node.soft_get('can_be_fused'):
                return False, 'Current node "{}" and reference node "{}" have different "can_be_fused" parameter ' \
                              '{} and {}'.format(node.id, node_ref.id, node.soft_get('can_be_fused'),
                                                 node_ref.soft_get('can_be_fused'))

        if node_ref.kind == 'op':
            # Check that nodes has same operation
            if check_op_attrs:
                for attr in graph_ref.node[node_ref.id]:
                    if graph_ref.node[node_ref.id][attr] is None or attr in ['name', 'id', '_in_ports', '_out_ports',
                                                                             'infer', 'IE']:
                        continue
                    if attr not in graph.node[node.id]:
                        return False, 'Current node "{}" has missing attribute {}'.format(node.id, attr)

                    if type(graph_ref.node[node_ref.id][attr]) in [np.ndarray, list]:
                        if not np.array_equal(graph.node[node.id][attr], graph_ref.node[node_ref.id][attr]):
                            return False, 'Current node "{}" and reference node "{}" have different attr "{}" : ' \
                                          '{} and {}'.format(node.id, node_ref.id, attr, graph.node[node.id][attr],
                                                             graph_ref.node[node_ref.id][attr])
                    elif isinstance(graph.node[node.id][attr], Number):
                        eps = 5e-2 if node.has('precision') and node['precision'] == 'FP16' else 1e-4
                        if abs(graph.node[node.id][attr] - graph_ref.node[node_ref.id][attr]) > eps:
                            return False, '{} and {} has different attr {} : {} and {}'.format(
                                node.id, node_ref.id, attr, graph.node[node.id][attr],
                                graph_ref.node[node_ref.id][attr])
                    elif graph.node[node.id][attr] != graph_ref.node[node_ref.id][attr]:
                        return False, 'Current node "{}" and reference node "{}" have different attr "{}" : {} and {}'.format(
                            node.id, node_ref.id, attr, graph.node[node.id][attr],
                            graph_ref.node[node_ref.id][attr])

        else:
            if node_ref.has_valid('shape') and not node.has_valid('shape'):
                return False, '{} has None shape'.format(node.id)
            if node_ref.has_valid('value') and not node.has_valid('value'):
                return False, '{} has None value'.format(node.id)

            # Check that nodes has same shape and value
            if node_ref.has_valid('shape') and node_ref.shape is not None and not np.array_equal(node_ref.shape,
                                                                                                 node.shape):
                return False, 'Current node "{}" and reference node "{}" have different shapes {} and {}'.\
                    format(node.id, node_ref.id, node.shape, node_ref.shape)

            if node_ref.has_valid('value') and node_ref.value is not None:
                eps = 5e-2 if np.asarray(node.value).dtype == 'float16' else 1e-4
                if not np.allclose(node_ref.value, node.value, rtol=eps, atol=eps):
                    return False, 'Current node "{}" and reference node "{}" have different values \n{} \nand \n{}'.\
                        format(node.id, node_ref.id, node.value, node_ref.value)
        ports = sorted(node.in_nodes().keys()) if node.kind == 'op' else None
        in_nodes = [node.in_node(k) for k in ports] if node.kind == 'op' else node.in_nodes()
        for in_node in in_nodes:
            if in_node.id not in checked_nodes and in_node.id not in q:
                q.append(in_node.id)

        ports_ref = sorted(node_ref.in_nodes().keys()) if node_ref.kind == 'op' else None
        if ports != ports_ref:
            return False, 'Current node "{}" and reference node "{}" have different ports'.format(node.id, node_ref.id)

        in_nodes = [node_ref.in_node(k) for k in ports] if node_ref.kind == 'op' else node_ref.in_nodes()
        for in_node in in_nodes:
            if in_node.id not in checked_nodes_ref and in_node.id not in q_ref:
                q_ref.append(in_node.id)

        out_nodes = node.out_nodes().values() if node.kind == 'op' else node.out_nodes()
        for out_node in out_nodes:
            if out_node.id not in checked_nodes and out_node.id not in q:
                q.append(out_node.id)

        out_nodes = node_ref.out_nodes().values() if node_ref.kind == 'op' else node_ref.out_nodes()
        for out_node in out_nodes:
            if out_node.id not in checked_nodes_ref and out_node.id not in q_ref:
                q_ref.append(out_node.id)

    return True, ''


class FakeNode:
    def __init__(self, pl, ml):
        self.pb = pl
        self.model_pb = ml
        self.graph = None
        self.update_node = lambda: None

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)
