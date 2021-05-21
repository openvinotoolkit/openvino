# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.back.pass_separator import BackFinish
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph, rename_node


class NamesUniquenessCheck(BackReplacementPattern):
    enabled = True

    def run_after(self):
        return [BackFinish]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        names_dict = dict()
        not_unique_names = set()
        for node in graph.get_op_nodes():
            node_name = node.soft_get('name', node.id)
            if node_name not in names_dict.keys():
                names_dict[node_name] = [node]
            else:
                names_dict[node_name].append(node)
                not_unique_names.add(node_name)

        if not_unique_names:
            for name in not_unique_names:
                for idx, node in enumerate(names_dict[name]):
                    if node.op is not "Result":
                        rename_node(node, name + '_{}'.format(idx))
