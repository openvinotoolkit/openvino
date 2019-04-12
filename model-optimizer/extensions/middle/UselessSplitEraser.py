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

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern


class UselessSplitEraser(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import PreMiddleStart
        return [PreMiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[('split', {'kind': 'op', 'op': 'Split', 'num_split': 1})],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        split_node = match['split']
        input = split_node.in_node(1)
        output = split_node.out_node()
        graph.remove_edge(input.id, split_node.id)

        for u, v, d in list(graph.out_edges(output.id, data=True)):
            graph.add_edges_from([(input.id, v, d)])
            graph.remove_edge(u, v)
