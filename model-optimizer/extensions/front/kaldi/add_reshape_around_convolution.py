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
from mo.ops.reshape import Reshape


class ReplaceConvolutionReshape(FrontReplacementSubgraph):
    """
       This pass adds Reshapes around a Convolution layer for reshaping from NH to NCHW
       For example:
           Let's suppose we have next graph:

           Prev_Layer [N, H] -> Convolution [N, C, H, W] -> Next_Layer [N, H]

           In this case Convolution takes only [N, H] from input tensor in 3rd dim
           So this pass will convert this graph to the next one:

           Prev_Layer [N, H] -> Reshape [N, 1, H, 1] -> Convolution [N, C=1, H, W=1] -> Reshape [N, 1, H, 1] -> Next_Layer [N, H]

   """
    enabled = True

    def pattern(self):
        return dict(nodes=[('conv', dict(op='Convolution'))],
                    edges=[])

    def replace_sub_graph(self, graph: Graph, match: dict):
        node = match['conv']
        input_reshape_node = Reshape(graph,
                                     {
                                         'name': '/Reshape/' + node.name,
                                         'axis': 1,
                                         'infer': Reshape.kaldi_infer
                                     }).create_node()
        output_reshape_node = Reshape(graph,
                                      {
                                          'name': node.name + '/Reshape/',
                                          'axis': 1,
                                          'infer': Reshape.kaldi_infer
                                      }).create_node()
        # connect input_reshape_node
        source = node.in_port(0).get_source()
        node.in_port(0).get_connection().set_source(input_reshape_node.out_port(0))
        input_reshape_node.in_port(0).connect(source)
        # connect output_reshape_node
        node.out_port(0).get_connection().set_source(output_reshape_node.out_port(0))
        node.out_port(0).connect(output_reshape_node.in_port(0))
