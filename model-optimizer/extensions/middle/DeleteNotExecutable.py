"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class DeleteNotExecutable(MiddleReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.middle.TensorIteratorConditionChecker import ConditionChecks
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
