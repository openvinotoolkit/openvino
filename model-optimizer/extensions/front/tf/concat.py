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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Graph


class Concat(FrontReplacementSubgraph):
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('concat', dict(op='Concat', simple_concat=True))],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        """
        There are Concat and ConcatV2 operations in TensorFlow
        The main difference is incoming port of tensor representing axis of concatenation
        In Concat it is the 0 port, in ConcatV2 it is the last port
        To reuse ConcatV2 logic (infer) that already exists in the Model Optimizer here we renumber ports of Concat
        """
        in_edges = list(graph.in_edges(match['concat'].id, data=True))
        for u, v, attrs in in_edges:
            in_port = attrs['in']
            attrs['in'] = len(in_edges) - 1 if in_port == 0 else attrs['in'] - 1
        if match['concat'].has('axis'):
            # we delete axis parameter here (it was set by default by Concat Op) to carefully get it from the last
            # input in Concat infer function
            del graph.node[match['concat'].id]['axis']
