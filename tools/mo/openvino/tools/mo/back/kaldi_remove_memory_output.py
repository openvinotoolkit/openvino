# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class KaldiRemoveMemoryOutputBackReplacementPattern(BackReplacementPattern):
    enabled = True

    def run_after(self):
        from openvino.tools.mo.back.pass_separator import BackFinish
        return [BackFinish]

    def run_before(self):
        from openvino.tools.mo.back.SpecialNodesFinalization import CreateConstNodesReplacement
        return [CreateConstNodesReplacement]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('memory_node', dict(op='Assign')),
                ('data_node', dict(kind='data')),
                ('op_output', dict(op='Result'))
            ],
            edges=[
                ('memory_node', 'data_node'),
                ('data_node', 'op_output')
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Need to find the pattern: Memory -> Data -> Result

        It is needed to make Memory nodes appear in IR,
        but they are output nodes by default and we remove the Result node after each output memory.

        DO NOT use graph clean up after it
        otherwise Memory nodes would be removed as they are not on the path from input to output

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
        match : dict
           Patterns which were found in graph structure.
        """
        memory = match['memory_node']
        data = match['data_node']
        op_output = match['op_output']

        graph.remove_edge(memory.id, data.id)
        graph.remove_node(data.id)
        graph.remove_node(op_output.id)
