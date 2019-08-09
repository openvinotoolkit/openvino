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

import networkx as nx

from mo.graph.graph import Node, Graph


def find_nodes_by_attribute_value(graph: Graph, attr: str, attr_name: str):
    return [id for id, v in nx.get_node_attributes(graph, attr).items() if v == attr_name]


def find_inputs(graph: Graph):
    return find_nodes_by_attribute_value(graph, 'type', 'Parameter')


def find_outputs(graph: Graph):
    outputs = []
    for node_id in find_nodes_by_attribute_value(graph, 'op', 'Result'):
        parents = Node(graph, node_id).in_nodes()
        assert len(parents) == 1, 'Result node should have exactly one input'
        parent = parents[0].id
        outputs.append(parent)
    return list(set(outputs))
