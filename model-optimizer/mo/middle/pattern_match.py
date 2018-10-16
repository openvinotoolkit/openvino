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

import networkx as nx
from networkx.algorithms import isomorphism as ism

from mo.graph.graph import Node


def inverse_dict(d: dict):
    return {v: k for k, v in d.items()}


def apply_pattern(graph: nx.MultiDiGraph, nodes: list, edges: list, action: callable, node_attrs: list,
                  edge_attrs: list):
    """
    Search for all matches of a given subgraph defined by [nodes, edges] in graph,
    then apply action for each such match.
    """
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
    return all(data1.get(attr, None) == data2.get(attr, None) for attr in data2.keys())


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


def find_pattern_matches(graph: nx.MultiDiGraph, nodes: list, edges: list, node_attrs: list, edge_attrs: list):
    """
    Find all matches of a given sub-graph defined by [nodes, edges] in graph.
    """
    subgraph = nx.MultiDiGraph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(edges)
    matcher = ism.MultiDiGraphMatcher(graph, subgraph, node_match, edge_match)
    return matcher.subgraph_isomorphisms_iter()
