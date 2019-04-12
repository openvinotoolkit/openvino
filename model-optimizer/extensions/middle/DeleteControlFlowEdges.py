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


class DeleteControlFlowEdges(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.PartialInfer import PartialInfer
        return [PartialInfer]

    def run_before(self):
        return []

    def find_and_replace_pattern(self, graph: Graph):
        for u, v, k, attrs in list(graph.edges(keys=True, data=True)):
            if 'control_flow_edge' in attrs and attrs['control_flow_edge']:
                graph.remove_edge(u, v, k)
                log.debug('Removing control flow edge from {} to {}'.format(u, v))
