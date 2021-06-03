# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.back.pass_separator import BackFinish
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, rename_node, Node


def update_dictionaries(names_dict: dict, not_unique_names: set, node: Node, node_name: str):
    if node_name not in names_dict.keys():
        names_dict[node_name] = [node]
    else:
        names_dict[node_name].append(node)
        not_unique_names.add(node_name)


def rename_not_unique_nodes(names_dict_1: dict, names_dict_2: dict, not_unique_names: set, skip_result: bool = True):
    for name in not_unique_names:
        for idx, node in enumerate(names_dict_1[name]):
            new_node_name = name + '_{}'.format(idx)
            # preparing a new unique name for the node
            while new_node_name in names_dict_1.keys() or new_node_name in names_dict_2.keys():
                new_node_name += '_{}'.format(idx)

            if skip_result:
                if node.soft_get('op') is not 'Result':
                    rename_node(node, new_node_name)
            else:
                rename_node(node, new_node_name)


class NamesUniquenessCheck(BackReplacementPattern):
    """
    If there are several layers with the same name in the original model and they are saved in the IR, IE will fail with
    the invalid IR error. IE checks the uniqueness of the names and, if it is not true, throws an exception. The way how
    to fix it on the MO side is to rename this nodes. Since we prefer to save framework names for the output nodes,
    nodes with op=Result will not be renamed, except the case when there are several Result nodes with the same name.
    """
    enabled = True

    def run_after(self):
        return [BackFinish]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        names_dict = dict()
        not_unique_names = set()
        results = dict()
        not_unique_results = set()
        for node in graph.get_op_nodes():
            node_name = node.soft_get('name', node.id)
            update_dictionaries(names_dict, not_unique_names, node, node_name)
            if node.soft_get('op') is 'Result':
                update_dictionaries(results, not_unique_results, node, node_name)

        if not_unique_names:
            rename_not_unique_nodes(names_dict, results, not_unique_names)
        if not_unique_results:
            rename_not_unique_nodes(results, names_dict, not_unique_results, skip_result=False)
