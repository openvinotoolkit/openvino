# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from extensions.back.pass_separator import BackFinish
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, rename_node


def nodes_with_equal_names(graph: Graph):
    """
    :param graph: Graph to operate on
    :return: Dictionary with node names as keys and a list of their corresponding nodes as values
    """
    names_dict = defaultdict(list)
    for node in graph.get_op_nodes():
        node_name = node.soft_get('name', node.id)
        names_dict[node_name].append(node)
    return names_dict


def make_node_names_unique(nodes: list, node_names: set):
    """
    :param nodes: List with nodes matching a specific name
    :param node_names: Set with all node names contained in the graph
    :return: None

    Result nodes will be renamed only when it is absolutely necessary(if there are several Result nodes with the same name).
    Function finds a position of Result nodes in the "nodes" list, take the first and rename all other nodes.
    If the "nodes" list does not contain Result nodes, then all nodes starting from the second one will be renamed.
    All new names are added to the "node_names" set.
    """
    results_pos = [idx for idx, node in enumerate(nodes) if node.op == 'Result']
    node_position_to_keep = 0
    if len(results_pos) != 0:
        node_position_to_keep = results_pos[0]
    for idx, node in enumerate(nodes):
        if idx != node_position_to_keep:
            new_node_name = node.soft_get('name', node.id) + '_' + str(idx)
            # preparing a new unique name for the node
            while new_node_name in node_names:
                new_node_name += '_' + str(idx)
            node_names.add(new_node_name)
            rename_node(node, new_node_name)


class NamesUniquenessCheck(BackReplacementPattern):
    """
    If there are several layers with the same name in the original model and they are saved in the IR, IE will fail with
    the invalid IR error. IE checks the uniqueness of the names and, if it is not true, throws an exception. The way how
    to fix it on the MO side is to rename this nodes (one node will remain with the original name). Since we prefer to
    save framework names for the output nodes, nodes with op=Result will not be renamed, except the case when there are
    several Result nodes with the same name.
    """
    enabled = True

    def run_after(self):
        return [BackFinish]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        names_to_nodes = nodes_with_equal_names(graph)
        node_names = set(names_to_nodes.keys())
        for nodes in names_to_nodes.values():
            if len(nodes) > 1:
                make_node_names_unique(nodes, node_names)
