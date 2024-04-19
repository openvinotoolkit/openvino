# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph, Node


class AddInputDataToPriorBoxes(FrontReplacementPattern):
    enabled = True

    def run_before(self):
        from openvino.tools.mo.front.create_tensor_nodes import CreateTensorNodes
        return [CreateTensorNodes]

    def run_after(self):
        from openvino.tools.mo.front.pass_separator import FrontFinish
        return [FrontFinish]

    @staticmethod
    def add_input_data_to_prior_boxes(graph: Graph, input_names: str = ''):
        """
        PriorBox layer has data input unlike mxnet.
        Need to add data input to _contrib_MultiBoxPrior for
        for correct conversion to PriorBox layer.

        Parameters
        ----------
        graph : Graph
           Graph with loaded model.
        """
        if not input_names:
            input_names = ('data',)
        else:
            input_names = input_names.split(',')

        input_nodes = {}
        for node in graph.nodes():
            node = Node(graph, node)
            if node.has_valid('op') and node.name in input_names:
                input_nodes.update({node.id: node})

        if len(input_nodes) > 0:
            for node in graph.nodes():
                node = Node(graph, node)
                if node.has_valid('op') and node.op == '_contrib_MultiBoxPrior':
                    node.add_input_port(idx=1)
                    graph.create_edge(list(input_nodes.values())[0], node, out_port=0, in_port=1)

    def find_and_replace_pattern(self, graph: Graph):
        self.add_input_data_to_prior_boxes(graph, graph.graph['cmd_params'].input)
