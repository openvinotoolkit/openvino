"""
 Copyright (C) 2020 Intel Corporation

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
from extensions.back.FuseTransposesSequence import FuseTransposesSequence
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.back.replacement import BackReplacementPattern


class ConvolutionWithDynamicWeightsTranspose(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == 'tf']
    force_shape_inference = True

    def run_before(self):
        return [FuseTransposesSequence]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='Convolution'):
            weights_node = node.in_port(1).get_source().node
            if weights_node.soft_get('type') != 'Const' and weights_node.soft_get('type') != 'FakeQuantize':
                transpose = create_op_node_with_second_input(graph, Transpose, int64_array([3, 2, 0, 1]), {'override_output_shape': True},
                                                             input_node=node.in_port(1).get_source().node)
                weights_node.out_port(0).get_connection().insert_node(transpose)

