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
from collections import deque

import numpy as np

from extensions.front.kaldi.add_reshape_around_convolution import ReplaceConvolutionReshape
from extensions.middle.TensorIteratorMerge import op_type
from extensions.ops.activation_ops import activation_ops
from extensions.ops.transpose import Transpose
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.graph.graph import Node, Graph
from mo.ops.const import Const


class ReplaceConvolutionTranspose(FrontReplacementSubgraph):
    """
    This pass adds Transpose around a Convolution layer if after there is sequence Pooling or Activation afterConvolution
    **IMPORTANT**: This pass must run after inserting Reshapes around Poolings and Convolutions
       For example:
           Let's suppose we have next graph:

           Convolution -> [Pooling | Activation -> Pooling | Pooling -> Activation | Activation]* -> ... -> (ScaleShift | FullyConnected)

           **NOTE**: Please, remember about Reshapes around Poolings and Convolutions.
                     In this example we do not print them for simplicity.
           **NOTE**: After Convolution, it is not necessary to have a sequence [Pooling | Activation -> Pooling | Pooling -> Activation | Activation]*

           So this pass will convert this graph to the next one:

           Convolution -> * -> Transpose (order 0, 3, 2, 1 )-> Next_Layer -> ... -> (ScaleShift|FullyConnected)

    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('target_node', dict(op=lambda x: x in ['ScaleShift', 'FullyConnected']))
            ],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: dict):
        target_node = match['target_node']
        nodes_with_weights = self.dfs(graph, target_node.name, ('Convolution', 'FullyConnected', 'ScaleShift'), True)
        convolution_nodes = [node for node in nodes_with_weights if Node(graph, node).op == 'Convolution']
        for convolution_node in convolution_nodes:
            target_node = self.search_target_node(Node(graph, convolution_node))
            order_const = Const(graph, dict(value=np.array([0, 3, 2, 1]))).create_node()
            permute_node = Transpose(graph, dict(name=target_node.name + '/Transpose')).create_node()
            target_node.insert_node_after(permute_node, 0)
            order_const.out_port(0).connect(permute_node.in_port(1))

    def run_after(self):
        from extensions.front.kaldi.add_reshape_around_pooling import ReplacePoolingReshape
        return [ReplaceConvolutionReshape, ReplacePoolingReshape]

    @staticmethod
    def search_target_node(node: Node):
        target_node = ReplaceConvolutionTranspose.skip_reshapes(node)
        sequence_layers = [['Pooling'], activation_ops]
        if target_node.op not in ['Pooling'] + activation_ops:
            return node
        if target_node.op in activation_ops:
            sequence_layers.reverse()
        if target_node.op in sequence_layers[0]:
            next_node = ReplaceConvolutionTranspose.skip_reshapes(target_node)
            if next_node.op in sequence_layers[1]:
                target_node = next_node
        return target_node

    @staticmethod
    def skip_reshapes(node: Node):
        next_node = node.out_node()
        while next_node.op == 'Reshape':
            next_node = next_node.out_node()
        return next_node

    @staticmethod
    def dfs(graph: Graph, node_name: str, stop_nodes: tuple, reverse: bool = False) -> list:
        d = deque()
        res = []
        visited = set()
        visited.add(node_name)
        d.appendleft(node_name)
        while len(d) != 0:
            cur_node = d.popleft()
            if reverse:
                nodes = graph.in_edges(cur_node)
            else:
                nodes = graph.out_edges(cur_node)
            for in_node_name, _ in nodes:
                if in_node_name not in visited:
                    if op_type(graph, in_node_name) not in stop_nodes:
                        visited.add(in_node_name)
                        d.append(in_node_name)
                    else:
                        res.append(in_node_name)
        return res
