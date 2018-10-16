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

import networkx as nx

from mo.graph.graph import NodeWrap


def find_nodes_by_type(graph: nx.MultiDiGraph, t_name: str):
    nodes = nx.topological_sort(graph)
    inputs = []
    for n in nodes:
        node = NodeWrap(graph, n)
        if node.has('type') and node.type == t_name:
            inputs.append(node.id)
    return inputs


def find_inputs(graph: nx.MultiDiGraph):
    return find_nodes_by_type(graph, 'Input')


def find_outputs(graph):
    nodes = nx.topological_sort(graph)
    outputs = []
    for n in nodes:
        node = NodeWrap(graph, n)
        if node.has('is_output') and node['is_output']:
            outputs.append(node.id)
    return outputs
