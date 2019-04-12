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

from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.pooling import Pooling
from mo.ops.reshape import Reshape


class ReplacePoolingReshape(FrontReplacementOp):
    """
        This pass adds Reshapes around a Pooling layer for reshaping from NH to NCHW
        For example:
            Let's suppose we have next graph:

            Prev_Layer [N, H] -> Pooling [N, C, H, W] -> Next_Layer [N, H]

            In this case Pooling takes only [N, H] from input tensor in 3rd dim
            So this pass will convert this graph to the next one:

            Prev_Layer [N, H] -> Reshape [N, 1, H, 1] -> Pooling [N, C=1, H, W=1] -> Reshape [N, 1, H, 1] -> Next_Layer [N, H]

    """
    op = "Pooling"
    enabled = True

    def replace_op(self, graph: Graph, node: Node) -> list:
        input_node = node.in_node(0)

        input_reshape_node = Reshape(graph,
                                     {
                                         'name': 'Reshape/' + node.name,
                                         'infer': Reshape.kaldi_infer
                                     }).create_node([input_node])

        pooling_node = Pooling(graph, graph.node[node.id]).create_node([input_reshape_node])

        output_reshape_node = Reshape(graph,
                                      {
                                          'name': node.name + '/Reshape/',
                                          'infer': Reshape.kaldi_infer
                                      }).create_node([pooling_node])

        return [output_reshape_node.id]
