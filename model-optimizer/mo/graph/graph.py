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

import collections
from copy import deepcopy
import logging as log

import networkx as nx
import numpy as np

from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def unique_id(graph: nx.MultiDiGraph, prefix: str = ""):
    """
    Generates a unique node id for a new node in a given graph.
    The optional string prefix can be specified.
    """
    # TODO thread safety?
    unique_id.count = max(unique_id.count, graph.number_of_nodes()) + 1
    if prefix and not graph.has_node(prefix):
        return str(prefix)
    while graph.has_node(prefix + str(unique_id.count)):
        unique_id.count += 1
    return prefix + str(unique_id.count)


unique_id.count = 0


def get_node_id_by_name(graph: nx.MultiDiGraph, name: str):
    for node in graph.nodes():
        if 'name' in graph.node[node] and graph.node[node]['name'] == name:
            return node
    raise Error('No node with name {}. ' +
                refer_to_faq_msg(51), name)


def create_graph_with_nodes(src_nodes, get_id: callable, get_attrs: callable):
    """
    Go over all nodes in src_nodes that should be enumerable and create new NX nodes
    using get_id and get_attrs functions to create node id and node attributes correspondingly.
    """
    graph = nx.MultiDiGraph()
    for node in src_nodes:
        graph.add_node(get_id(node), **get_attrs(node))
    return graph


# TODO implement merging for keys with dictionary values?
def merge_edge_props(attrs: dict, additional_attrs: dict):
    """
    Update edge attributes without changing 'in' and 'out' keys.
    It is necessary to copy edge attributes during merging of nodes when
    result of one subgraph call is passed as input to another subgraph call
    """
    result = attrs
    for (key, value) in additional_attrs.items():
        if key not in ['in', 'out']:
            if type(additional_attrs[key]) is list:
                if key not in result:
                    result[key] = []
                result[key].extend(additional_attrs[key])
                result[key] = list(set(result[key]))  # silly solution to find unique elements
            else:
                result[key] = value
    return result


def print_graph_stat(graph: nx.MultiDiGraph):
    log.debug('Number of nodes in graph: {}'.format(graph.number_of_nodes()))
    log.debug('Number of edges in graph: {}'.format(len(list(graph.edges()))))
    ops = collections.defaultdict(int)
    for _node in graph.nodes():
        node = NodeWrap(graph, _node)
        kind = node.kind if node.has('kind') else '<UNDEFINED>'
        if node.has('op'):
            ops['op/' + node.op] += 1
        else:
            ops[kind] += 1
        if node.has('shape') and np.any(node.shape == 0):
            log.error("Found bad shape: '{}' for node '{}'".format(node.shape, node.node))
    for k, v in ops.items():
        log.debug('   {} : {}'.format(k, v))


def get_inputs_with_ports(graph, match, pattern_edges, input_names_in_pattern):
    """
    Front replacements of multi-input nodes should specify output port to add_node-like functions
    This function is a helper to get such information out of matched nodes
    :param graph: graph to operate on
    :param match: dictionary returned by matching function
    :param pattern_edges: edges that are specified in pattern
    :param input_names_in_pattern: names of matched nodes as they were specified in pattern that should be in
    resulting list
    :return: list of tuples of node and output port
    """
    inputs = []
    for name in input_names_in_pattern:
        assert name in match, "node named {} not in match {}".format(name, match)
        src = match[name]
        dst = []
        for edge in pattern_edges:
            if edge[0] == name:
                assert edge[1] in match, "name from pattern_edges {} not in match {}".format(edge[1], match)
                dst.append(match[edge[1]])
        if len(dst) != 1:
            raise Error('Multiple output ports detected for node {} as {} in pattern'.format(match[name].id, name))
        dst = dst[0]
        out_port = graph.get_edge_data(src.id, dst.id)[0]['out']
        inputs.append((src, out_port))
    return inputs


def dump_graph_for_graphviz(graph: nx.MultiDiGraph, node_attrs: list = ['kind', 'op', 'shape'],
                            edge_attrs: list = ['in', 'out'],
                            nodes_to_dump: list = None, save_to_svg = False):
    log.debug("---- GRAPHVIZ OUTPUT STARTS ----")
    if nodes_to_dump is None:
        nodes_to_dump = graph.nodes()
    string = '\ndigraph {\n'
    visited_nodes = set()
    for src_node_name, dst_node_name, attrs in graph.edges(data=True):
        visited_nodes.add(src_node_name)
        visited_nodes.add(dst_node_name)
        if src_node_name not in nodes_to_dump or dst_node_name not in nodes_to_dump:
            continue
        src_node = graph.node[src_node_name]
        dst_node = graph.node[dst_node_name]
        src_node_string = str(src_node_name) + '\\n' + '\\n'.join(
            [str(key) + '=' + str(src_node.get(key, 'None')) for key in node_attrs if key in src_node])
        dst_node_string = str(dst_node_name) + '\\n' + '\\n'.join(
            [str(key) + '=' + str(dst_node.get(key, 'None')) for key in node_attrs if key in dst_node])
        edge_string = ' '.join([str(key) + '=' + str(attrs.get(key, 'None')) for key in edge_attrs if key in attrs])
        string += '"{}" -> "{}" [label = "{}"];\n'.format(src_node_string, dst_node_string, edge_string)
    for node in nodes_to_dump:
        if node not in visited_nodes:
            string += '"{}"'.format(node) # TODO: add attributes like it was done in the loop above
            visited_nodes.add(node)
    string += '}'
    log.debug(string)
    log.debug("---- GRAPHVIZ OUTPUT ENDS ----")

    if save_to_svg:
        try:
            import graphviz
            import os
            file_name = "{}_{}.txt".format(graph.name.replace('/', '_'), 0)
            id = 1
            while os.path.exists(file_name):
                file_name = "{}_{}.txt".format(graph.name.replace('/', '_'), id)
                id += 1
            with open(file_name, "w") as f:
                f.write(string)
            graphviz.render('dot','svg', file_name)
            print('Graph was saved to {}.{}'.format(file_name, 'svg'))
        except ImportError:
            raise ImportError('Can\'t import graphviz')
        except Exception as e:
            raise Error('Can\'t save graph to svg') from e

    return string


def create_sub_graph_copy(graph: nx.MultiDiGraph, nodes_to_extract: list):
    """
    Create new graph which is a sub-graph of the 'graph' that contains just nodes from 'nodes_to_extract' list. The
    returned sub-graph is a deep copy of the provided graph nodes.
    :param graph: graph to create a sub-graph from.
    :param nodes_to_extract: list of node names to extract.
    :return: new graph.
    """
    return graph.subgraph(nodes_to_extract).copy()


def get_inputs(graph: nx.MultiDiGraph, node: str, edge_attr: dict = {}, control_flow: bool = False):
    in_edges = graph.in_edges(node, data=True)
    if not control_flow:
        in_edges = [(u, v, d) for u, v, d in in_edges if 'control_flow_edge' not in d or not d['control_flow_edge']]
    return [(u, d) for u, v, d in in_edges if all([attr in d and d[attr] == edge_attr[attr] for attr in edge_attr])]


def get_outputs(graph: nx.MultiDiGraph, node: str, edge_attr: dict = {}, control_flow: bool = False):
    out_edges = graph.out_edges(node, data=True)
    if not control_flow:
        out_edges = [(u, v, d) for u, v, d in out_edges if 'control_flow_edge' not in d or not d['control_flow_edge']]
    return [(v, d) for u, v, d in out_edges if all([attr in d and d[attr] == edge_attr[attr] for attr in edge_attr])]


def get_single_input(graph: nx.MultiDiGraph, node: str, edge_attr: dict = {}):
    """
    Searches for all edges that have given attributes.
    If there no such edges or there are multiple edges, raise exception.
    If there is only one edge, returns the source node for this edge
    and the edge attributes themselves.
    """
    inputs = get_inputs(graph, node, edge_attr)
    if len(inputs) != 1:
        log.debug("Node '{}' has {} inputs with edge attributes '{}'".format(node, inputs, str(edge_attr)))
        raise AttributeError(
            "None or multiple inputs satisfy given attributes. Node: " + str(node) + ", edge_attr: " + str(edge_attr))
    return inputs[0]


def get_single_output(graph: nx.MultiDiGraph, node: str, edge_attr: dict = {}):
    outputs = get_outputs(graph, node, edge_attr)
    if len(outputs) != 1:
        log.debug("Node '{}' has {} outputs with edge attributes '{}'".format(node, outputs, str(edge_attr)))
        raise AttributeError(
            "None or multiple outputs satisfy given attributes. Node: " + str(node) + ", edge_attr: " + str(edge_attr))
    return outputs[0]


def get_graph_ops(graph: nx.MultiDiGraph):
    return [Node(graph, node) for node in graph.nodes() if Node(graph, node).soft_get('kind') == 'op']


def dict_includes(big: dict, sub_dict: dict):
    ''' Searches attributes from sub_dict in big and ensures that all values match. '''
    return all(big.get(attr, None) == sub_dict.get(attr, None) for attr in sub_dict.keys())


class NodeWrap:

    def __init__(self, graph: nx.MultiDiGraph, node: str):
        super(NodeWrap, self).__setattr__('graph', graph)
        super(NodeWrap, self).__setattr__('node', node)  # obsolete
        super(NodeWrap, self).__setattr__('id', node)

    def __setattr__(self, k, v):
        # you can assign only existing attributes
        attrs = self.graph.node[self.node]
        if not k in attrs:
            raise AttributeError
        attrs[k] = v

    def __getattr__(self, k):
        # hope it raises AttributeError if k is not in the dict
        return self.graph.node[self.node][k]

    def attrs(self):
        return self.graph.node[self.node]

    def has(self, k):
        return k in self.graph.node[self.node]

    def has_valid(self, k):
        return self.has(k) and not self.graph.node[self.node][k] is None

    def has_and_set(self, k):
        return self.has_valid(k) and self[k]

    def __getitem__(self, k):
        return self.graph.node[self.node][k]

    def __setitem__(self, k, v):
        self.graph.node[self.node][k] = v

    def __contains__(self, k):
        return self.has(k)

    def in_nodes_edges(self, control_flow: bool=False):
        return {x[1]['in']: (NodeWrap(self.graph, x[0]), x[1]) for x in get_inputs(self.graph, self.node, control_flow=control_flow)}

    def in_nodes(self, control_flow: bool=False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return {x[1]['in']: NodeWrap(self.graph, x[0]) for x in get_inputs(self.graph, self.node, control_flow=control_flow)}
        elif self.kind == 'data':
            return [NodeWrap(self.graph, n) for n, d in get_inputs(self.graph, self.node, control_flow=control_flow)]

    def in_edges(self, control_flow: bool=False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return {x[1]['in']: x[1] for x in get_inputs(self.graph, self.node, control_flow=control_flow)}
        elif self.kind == 'data':
            return [d for n, d in get_inputs(self.graph, self.node, control_flow=control_flow)]

    def out_nodes_edges(self, control_flow: bool=False):
        return {x[1]['out']: (NodeWrap(self.graph, x[0]), x[1]) for x in get_outputs(self.graph, self.node, control_flow=control_flow)}

    def out_nodes(self, control_flow: bool=False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return {x[1]['out']: NodeWrap(self.graph, x[0]) for x in get_outputs(self.graph, self.node, control_flow=control_flow)}
        elif self.kind == 'data':
            return [NodeWrap(self.graph, n) for n, d in get_outputs(self.graph, self.node, control_flow=control_flow)]

    def out_edges(self, control_flow: bool=False):
        assert self.has('kind')
        assert self.kind in ['op', 'data']
        if self.kind == 'op':
            return {x[1]['out']: x[1] for x in get_outputs(self.graph, self.node, control_flow=control_flow)}
        elif self.kind == 'data':
            return [d for n, d in get_outputs(self.graph, self.node, control_flow=control_flow)]

    def in_node(self, key=0, control_flow: bool=False):
        return self.in_nodes(control_flow=control_flow)[key]

    def out_node(self, key=0, control_flow: bool=False):
        return self.out_nodes(control_flow=control_flow)[key]

    def in_edge(self, key=0, control_flow: bool=False):
        return self.in_edges(control_flow=control_flow)[key]

    def out_edge(self, key=0, control_flow: bool=False):
        return self.out_edges(control_flow=control_flow)[key]

    def get_attrs(self):
        return self.graph.node[self.node]

    def soft_get(self, k):
        return self[k] if self.has_valid(k) else '<UNKNOWN>'

    def edges(self, attrs: dict=None):
        ''' Get a single edge with specified set of attributes.

            If none or multiple edges satisfies this criteria, exception is raised
            Edge is represented as tuple (u, v, d), where u is source node,
            v is destination node and d is edge attributes.
        '''
        edges = list(self.graph.in_edges([self.id], data=True)) + list(self.graph.out_edges([self.id], data=True))
        return [(u, v, d) for u,v,d in edges if dict_includes(d, attrs)]

    def edge(self, attrs: dict=None):
        ''' Get a single edge with specified set of attributes.

            If none or multiple edges satisfies this criteria, exception is raised
            Edge is represented as tuple (u, v, d), where u is source node,
            v is destination node and d is edge attributes.
        '''
        edges = self.edges(attrs)
        assert len(edges) == 1, 'edges: {}, required attributes: {}'.format(edges, attrs)
        return edges[0]

class Node(NodeWrap):
    pass


def get_sorted_inputs(node: Node, control_flow: bool=False):
    return sorted([x for x in get_inputs(node.graph, node.node, control_flow=control_flow) if 'in' in x[1]], key=lambda x: x[1]['in'])


def get_sorted_outputs(node: Node, control_flow: bool=False):
    return sorted([x for x in get_outputs(node.graph, node.node, control_flow=control_flow) if 'out' in x[1]], key=lambda x: x[1]['out'])


def create_edge(src_node: Node, dst_node: Node, out_port: int = 0, in_port: int = 0, edge_attrs: dict = None):
    """
    Creates edge from node 'src_node' from output with index 'out_port' to node 'dst_node' with input index 'in_port'.
    :param src_node: node to create edge from.
    :param dst_node: node to create edge to.
    :param out_port: the index of output tensor of the 'src_node'.
    :param in_port: the input index of the node 'dst_node'.
    :param edge_attrs: dictionary with edge attrs.
    :return: None
    """
    # edges must belong to the same graph
    assert src_node.graph is dst_node.graph
    graph = src_node.graph

    if edge_attrs is None:
        edge_attrs = dict()
    else:
        edge_attrs = edge_attrs.copy()
    edge_attrs.update({'in': in_port, 'out': out_port, 'in_attrs': ['in', 'permutation'], 'out_attrs': ['out', 'permutation'],
                       'data_attrs': ['fw_tensor_debug_info']})

    graph.add_edges_from([(src_node.id, dst_node.id, edge_attrs)])


def insert_node_after(node: Node, new_node: Node, node_out_port: int = 0):
    """
    Insert node 'new_node' after output with index 'node_out_port' of the node 'node'. All consumers of node 'node'
    output with index 'node_out_port' will be changed to consume node 'new_node'.
    The function should be used when graph doesn't contain data nodes yet.
    :param node: node after which new node should be inserted.
    :param new_node: node to be inserted.
    :param node_out_port: the output index for the node 'node' to insert
    :return: None
    """
    assert node.graph is new_node.graph
    assert (len([name for name in node.graph.nodes() if Node(node.graph, name).soft_get('kind') == 'data']) == 0)

    graph = node.graph
    old_edges = list(graph.out_edges(node.id, data=True, keys=True))
    # create new edges first and then remove all old edges. This is needed for case when 'node' has several consumers
    # getting input from 'node_out_port'.
    # save tuple ("name of the destination edge", "edge key") to be removed
    node_name_and_edge_key = []
    for _, dst_name, edge_key, edge_attrs in old_edges:
        if edge_attrs['out'] == node_out_port:
            log.debug('Create edge from "{}" to "{}"'.format(new_node.name, dst_name))
            create_edge(new_node, Node(graph, dst_name), 0, edge_attrs['in'])
            node_name_and_edge_key.append((dst_name, edge_key))
    for dst_name, edge_key in node_name_and_edge_key:
        log.debug('Remove edge from "{}" to "{}"'.format(node.id, dst_name))
        graph.remove_edge(node.id, dst_name, edge_key)
    create_edge(node, new_node, node_out_port, 0, {})


def erase_node(node: Node):
    """
    Erases node from the graph and reconnect edges from input node(s) to output node(s)
    Produces assertion error in case of multiple inputs and outputs node at the same time
    :param node: Node to erase
    """
    graph = node.graph
    node_id = node.id

    inputs = list(graph.in_edges(node_id, data=True))
    outputs = list(graph.out_edges(node_id, data=True))

    assert not (len(inputs) > 1 and len(outputs) > 1)

    if len(outputs) == 0 and len(inputs) != 0:
        for input, _, attrs in inputs:
            if Node(graph, node_id).has_and_set('is_output'):
                if graph.node[input]['kind'] == 'op':
                    data_nodes = [u for u, v in graph.in_edges(input)]
                    for data in data_nodes:
                        graph.node[data]['is_output'] = graph.node[node_id]['is_output']
                else:
                    graph.node[input]['is_output'] = graph.node[node_id]['is_output']

    if len(outputs) == 0 or len(inputs) == 0:
        graph.remove_node(node_id)
        return

    if len(outputs) == 1:
        output = outputs[0][1]
        for src, noop, attrs in inputs:
            graph.remove_edge(src, noop)
            graph.add_edge(src, output, **attrs)
        graph.remove_node(node_id)
        return

    if len(inputs) == 1:
        input = inputs[0][0]
        for noop, dst, attrs in outputs:
            graph.remove_edge(noop, dst)
            graph.add_edge(input, dst, **attrs)
        graph.remove_node(node_id)
        return


def replace_node(old_node: Node, new_node: Node, new_node_out_port: int=None):
    """
    Replaces node 'old_node' with a node 'new_node' preserving edge attributes.
    :param old_node: node to be replaced.
    :param new_node: node to replace with.
    :return: None
    """
    assert old_node.graph is new_node.graph
    graph = old_node.graph
    # save output edges and reconnect them to new node
    for _, dst_node_name, edge_attrs in graph.out_edges(old_node.id, data=True):
        new_edge_attrs = deepcopy(edge_attrs)
        if new_node_out_port is not None:
            assert 'out' not in edge_attrs or edge_attrs['out'] == 0, \
                'replace_node function can replace old node with a single output port only if new_node_out_port is specified'
            new_edge_attrs.update({'out': new_node_out_port})
        graph.add_edge(new_node.id, dst_node_name, **new_edge_attrs)

    # if the node for replace is output node then we propagate this attribute to a new node
    if old_node.has_valid('is_output') and old_node.is_output:
        old_node.is_output = False
        new_node['is_output'] = True
    graph.remove_node(old_node.id)


def check_empty_graph(graph: nx.MultiDiGraph, description: str):
    if len(graph.nodes()) <= 1:
        raise Error("Graph contains {} node after executing {}. It considered as error because resulting IR will be "
                    "empty which is not usual".format(len(graph.nodes()), description))


def copy_node(src_node: Node, new_attrs: dict=None, dst_graph: nx.MultiDiGraph=None):
    ''' Copies node with all attributes (optionally updated) within the same graph or to different graph.'''
    if new_attrs is None:
        new_attrs = {}
    if dst_graph is None:
        dst_graph = src_node.graph

    attrs = deepcopy(src_node.attrs())
    attrs.update(new_attrs)
    new_id = unique_id(dst_graph)
    dst_graph.add_node(new_id, attrs)
    return Node(dst_graph, new_id)
