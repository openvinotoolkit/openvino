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

import logging as log

import networkx as nx
from networkx.algorithms import isomorphism as ism

from mo.graph.graph import Node, dict_includes, Graph


def inverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def for_each_sub_graph(graph: Graph, func: callable):
    """ Run a given function `func` for each sub-graph in a given graph not recursively.

        It doesn't search for sub-graphs in found sub-graphs recursively. If the recursion is required,
        a given function `func` should be implemented in a special way to enable fully recursive traversal.
    """
    for node in graph.nodes():
        node = Node(graph, node)
        if node.has_valid('sub_graphs'):
            for sub_graph_name in node.sub_graphs:
                func(node[sub_graph_name])


def for_each_sub_graph_recursively(graph: Graph, func: callable):
    """ Run a given function `func` for each sub-graph in a given graph `graph` recursively.

        A given function `func` shouldn't contain a recursion for sub-graphs of the second level.
    """
    def recursive_helper(sub_graph):
        # user action
        func(sub_graph)
        # recursion
        for_each_sub_graph(sub_graph, recursive_helper)

    for_each_sub_graph(graph, recursive_helper)


def for_graph_and_each_sub_graph_recursively(graph: Graph, func: callable):
    """ Run a given function `func` for a given graph `graph` and each sub-graph recursively. """
    func(graph)
    for_each_sub_graph_recursively(graph, func)


def all_edges_in_nodes(nodes: list, edges: list):
    return all([edge[0] in nodes and edge[1] in nodes for edge in edges])


def apply_pattern(graph: Graph, nodes: list, edges: list, action: callable, node_attrs: list = None,
                  edge_attrs: list = None):
    """
    Search for all matches of a given subgraph defined by [nodes, edges] in graph,
    then apply action for each such match.
    """
    if not all_edges_in_nodes([node[0] for node in nodes], edges):
        log.warning("Incorrect pattern attributes: not all nodes from edges are in nodes. "
                    "Please, mention all nodes you need in pattern in nodes attribute. ")

    matches = []
    for match in find_pattern_matches(graph, nodes, edges, node_attrs, edge_attrs):
        matches.append(match)

    for match in matches:
        match = inverse_dict(match)
        still_valid = True
        for k in match:
            if not graph.has_node(match[k]):
                # Graph changed significantly
                still_valid = False
                log.warning("The graph has changed significantly during applying pattern:\n"
                            "nodes: {}\n"
                            "edges: {}\n"
                            "node_attrs: {}\n"
                            "edge_attrs: {}".format(nodes, edges, node_attrs, edge_attrs))
                break
            match[k] = Node(graph, match[k])
        if still_valid:
            action(graph, match)


def check_node_usages_out_of_match(match: dict, node_name_in_match_group: str):
    """
    Checks if node is consumed by nodes out of match
    :param match: dictionary with pattern match
    :param node_name_in_match_group: string
    :return:
    """
    assert node_name_in_match_group in match
    graph = match[node_name_in_match_group].graph
    all_node_ids = [match[name].id for name in match]
    in_out_node_ids = [u for u, _ in graph.in_edges(match[node_name_in_match_group].id)]
    in_out_node_ids.extend([v for _, v in graph.out_edges(match[node_name_in_match_group].id)])
    return all([n in all_node_ids for n in in_out_node_ids])


def node_match(data1: dict, data2: dict):
    # We have to skip _in_ports/_out_ports attributes for comparision as they are not comparable
    return dict_includes(data1, data2, skip_attr_names=['_in_ports', '_out_ports'])


def edge_match(datasets1, datasets2):
    attrs = list(datasets2[0].keys())
    values1 = set([])
    for data1 in datasets1.values():
        x = tuple(data1.get(attr, None) for attr in attrs)
        values1.add(x)
    values2 = set([])
    for data2 in datasets2.values():
        x = tuple(data2.get(attr, None) for attr in attrs)
        values2.add(x)
    return values1 == values2


def build_matcher(graph: Graph, nodes: list, edges: list, node_attrs: list = None,
                         edge_attrs: list = None):
    if node_attrs is not None or edge_attrs is not None:
        log.warning('\'edge_attrs\' or `\'node_attrs\'` parameter was passed to function \'find_pattern_matches\', '
                    'but they are not used anymore. Pattern matching proceeds according to \'nodes\' and \'edges\' '
                    'parameters. Please avoid passing \'edge_attrs\' and \'node_attrs\' parameters to any pattern '
                    'matching function like \'find_pattern_matches\', \'apply_pattern\' and \'pattern\' because it '
                    'will be deprecated in the next release.')

    subgraph = Graph(name='pattern')
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(edges)
    return ism.MultiDiGraphMatcher(graph, subgraph, node_match, edge_match)


def find_pattern_matches(graph: Graph, nodes: list, edges: list, node_attrs: list = None,
                         edge_attrs: list = None):
    """
    Find all matches of a given sub-graph defined by [nodes, edges] in graph.
    """
    matcher = build_matcher(graph, nodes, edges, node_attrs, edge_attrs)
    return matcher.subgraph_isomorphisms_iter()


def find_isomorphisms(graph: Graph, nodes: list, edges: list):
    ''' Find for isomorphism between a given graph and a pattern specified by a given nodes and edges.
        Applies the same rules as apply_pattern.
    '''
    matcher = build_matcher(graph, nodes, edges)
    result = []
    for match in matcher.isomorphisms_iter():
        match = inverse_dict(match)
        match = {k: Node(graph, match[k]) for k in match.keys()}
        result.append(match)
    return result
