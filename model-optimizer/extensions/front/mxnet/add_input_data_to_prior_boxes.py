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
from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph, Node


class AddInputDataToPriorBoxes(FrontReplacementPattern):
    enabled = True

    def run_before(self):
        from extensions.front.create_tensor_nodes import CreateTensorNodes
        return [CreateTensorNodes]

    def run_after(self):
        from extensions.front.pass_separator import FrontFinish
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
