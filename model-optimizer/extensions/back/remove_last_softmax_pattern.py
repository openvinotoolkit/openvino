"""
 Copyright (c) 2017-2018 Intel Corporation

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

from mo.back.replacement import BackReplacementPattern
from mo.middle.passes.eliminate import remove_op_node


class RemoveLastSoftMaxPattern(BackReplacementPattern):
    # This replacer is intentionally disabled and must be called if the flag --remove_output_softmax was enabled
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax_node', dict(kind='op', op='SoftMax'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        """
        Need to find the pattern: Parent (any type) -> SoftMAx -> OpOutput

        It is needed to remove output SoftMAx layer

        Parameters
        ----------
        graph : nx.MultiDiGraph
           Graph with loaded model.
        match : dict
           Patterns which were found in graph structure.
        """
        softmax = match['softmax_node']
        child = softmax.out_node()
        if not child.has_and_set('is_output'):
            return
        remove_op_node(graph, softmax)
