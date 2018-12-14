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
import numpy as np

from mo.graph.graph import Node
from mo.utils.dsu import DSU, DSUElem
from mo.utils.graph import bfs_search


def find_unsupported_ops(graph: nx.MultiDiGraph):
    """
    The function returns list of node name those are not supported. Currently nodes that product non FP32 data tensors
    or has undefined 'type' attribute are considered unsupported.
    :param graph: current graph with operations. Data nodes are not yet added.
    :return: the list of node names which are not supported
    """
    unsupported = list()
    for node_name in graph.nodes():
        node = Node(graph, node_name)
        # op node that produce non FP32 data or has no type are considered unsupported
        if node.kind == 'op':
            if not node.has_valid('type'):
                log.info('Node "{}" does not have type. Consider it unsupported'.format(node_name))
                unsupported.append(node.id)
            else:
                for out_data_node in node.out_nodes().values():
                    if out_data_node.has_valid('data_type') and out_data_node.data_type != np.float32:
                        log.info('Node "{}" produces output as non FP32. Consider it unsupported'.format(node_name))
                        unsupported.append(node.id)
    return unsupported


def find_unsupported_ops_subgraphs(graph: nx.MultiDiGraph, unsupported_nodes: list,
                                   find_constant_input_fn: callable = lambda node: node):
    bfs_nodes = bfs_search(graph, list())
    visited = set()
    # mark initial set of nodes as not supported
    for node_name in unsupported_nodes:
        graph.node[node_name]['supported'] = False

    for node_name in bfs_nodes:
        if node_name in visited:
            continue

        node = Node(graph, node_name)
        if node.has_valid('supported') and not node['supported']:
            added_nodes = find_constant_input_fn(node)
            visited.update(added_nodes)
            for node in added_nodes:
                node['supported'] = False

    dsu_elems = list()
    for node_name in bfs_nodes:
        node = Node(graph, node_name)
        if node.has_valid('supported') and not node['supported']:
            dsu_elems.append(DSUElem(node_name))

    dsu = DSU(dsu_elems)

    # merge adjacent unsupported nodes
    for dsu_elem in dsu_elems:
        node = Node(graph, dsu_elem.name)
        if not node['supported']:
            for out_node in node.out_nodes().values():
                if out_node.has_valid('supported') and not out_node['supported']:
                    dsu.union(dsu_elem, dsu.find_elem(out_node.id))

    subgraph_id = dict()  # key is the name of the node, value is the set of nodes that belong to this subgraph
    for dsu_elem in dsu.map.values():
        parent = dsu.find_parent(dsu_elem).name
        if parent not in subgraph_id.keys():
            subgraph_id[parent] = set()
        subgraph_id[parent].add(dsu_elem.name)

    return [list(s) for s in subgraph_id.values()]
