# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class DeleteNotExecutable(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from openvino.tools.mo.middle.TensorIteratorConditionChecker import ConditionChecks
        return [ConditionChecks]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        nodes_to_remove = set()
        for node_name, node_attrs in list(graph.nodes(data=True)):
            if node_attrs['kind'] == 'data' and 'executable' in node_attrs and not node_attrs['executable']:
                [nodes_to_remove.add(op) for op, _ in graph.in_edges(node_name)]
                nodes_to_remove.add(node_name)
        log.debug('Removing the following not executable nodes: {}'
                  ''.format('\n'.join(sorted(map(str, nodes_to_remove)))))
        graph.remove_nodes_from(nodes_to_remove)
