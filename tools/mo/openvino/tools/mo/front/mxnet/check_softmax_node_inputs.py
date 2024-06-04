# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph


class CheckSoftmaxNodeInputs(FrontReplacementPattern):
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.user_data_repack import UserDataRepack
        return [UserDataRepack]

    def run_after(self):
        return []

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('softmax', dict(op=lambda op: op in ['SoftMax', 'SoftmaxActivation', 'SoftmaxOutput']))
            ],
            edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Need to remove from softmax layer all unused inputs
        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
         match : dict
           Patterns which were found in graph structure.
        """
        softmax_node = match['softmax']
        softmax_nodes_len = len(softmax_node.in_nodes())
        for i in reversed(range(1, softmax_nodes_len)):
            in_node = softmax_node.in_node(i)
            graph.remove_edge(in_node.id, softmax_node.id)
            graph.remove_node(in_node.id)
