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


class KaldiRemoveMemoryOutputBackReplacementPattern(BackReplacementPattern):
    enabled = False

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('memory_node', dict(kind='op', op='Memory')),
                ('data_node', dict(kind='data'))
            ],
            edges=[
                ('memory_node', 'data_node', {'out': 0})
            ]
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        """
        Need to find the pattern: Memory -> Data -> OpOutput

        It is needed to make Memory nodes appear in IR,
        but they are output nodes by default and we remove the OpOutput node after each output memory.

        DO NOT use graph clean up after it
        otherwise Memory nodes would be removed as they are not on the path from input to output

        Parameters
        ----------
        graph : nx.MultiDiGraph
           Graph with loaded model.
        match : dict
           Patterns which were found in graph structure.
        """
        memory = match['memory_node']
        data = match['data_node']

        # Those Memory nodes that are not output ones, should not be replaced
        if not data.has_and_set('is_output'):
            return
        graph.remove_edge(memory.id, data.id)
        graph.remove_node(data.id)
