# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

from openvino.tools.mo.graph.graph import Node, Graph


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
