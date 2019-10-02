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
import math
import numpy as np

from extensions.ops.MatMul import MatMul
from extensions.ops.elementwise import Add, Mul
from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph
from mo.ops.reshape import Reshape


class FullyConnectedDecomposer(FrontReplacementSubgraph):
    """
     Decomposes FC operation:
         1. Biases are added separately with the help of Add node
         2. FC node itself is converted to MatMul
     """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', type='FullyConnected'))],
            edges=[]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['op']
        name = node.soft_get('name', node.id)

        # biases normalization
        if 2 in node.in_ports() and not node.in_port(2).disconnected():
            bias_node = Add(graph, {'name': name + '/Bias_'}).create_node()
            node.out_port(0).get_connection().set_source(bias_node.out_port(0))
            node.in_port(2).get_connection().set_destination(bias_node.in_port(1))
            node.out_port(0).connect(bias_node.in_port(0))

        # weights normalization
        assert node.has_valid('out-size')
        out_size = node['out-size']
        reshape_dim = int64_array([-1, out_size])
        if node.has_and_set('transpose_weights'):
            reshape_dim = int64_array([out_size, -1])
        node.insert_op_on_input_port(in_port_idx=1, new_op_class=Reshape,
                                     new_op_attrs={'name': name + '/weights_reshape'}, value=reshape_dim)
        if node.has_and_set('transpose_weights'):
            node.insert_op_on_input_port(in_port_idx=1, new_op_class=Transpose,
                                         new_op_attrs={'name': name + '/weights_transpose'}, value=int64_array([1, 0]))

        # input normalization for 4D Caffe and MxNet FullyConnected
        if graph.graph['fw'] in ['caffe', 'mxnet']:
            node.insert_op_on_input_port(in_port_idx=0, new_op_class=Reshape,
                                         new_op_attrs={'name': name + '/flatten_fc_input'}, value=int64_array([0, -1]))

        MatMul.update_node_stat(node, {})


class GemmDecomposer(FrontReplacementSubgraph):
    """
    Decomposes Gemm operation:
        1. Biases are added separately with the help of Add node
        2. Multiplication by `alpha` and `beta` values are separated to Mul operations
        3. Gemm operation itself is converted to MatMul
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[('op', dict(kind='op', op='Gemm'))],
            edges=[],
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):
        node = match['op']
        name = node.soft_get('name', node.id)

        # biases normalization
        bias_node = Add(graph, {'name': name + '/Bias_', 'can_be_scaleshift': False}).create_node()
        node.out_port(0).get_connection().set_source(bias_node.out_port(0))
        node.in_port(2).get_connection().set_destination(bias_node.in_port(1))
        node.out_port(0).connect(bias_node.in_port(0))

        if node.has_valid('alpha') and not math.isclose(node.alpha, 1):
            bias_node.insert_op_on_input_port(in_port_idx=0, new_op_class=Mul, value=np.array(node.alpha),
                                              new_op_attrs={'name': name + '/Alpha_', 'can_be_scaleshift': False})
            del node['alpha']

        if node.has_valid('beta') and not math.isclose(node.beta, 1):
            bias_node.insert_op_on_input_port(in_port_idx=1, new_op_class=Mul, value=np.array(node.beta),
                                              new_op_attrs={'name': name + '/Beta_', 'can_be_scaleshift': False})
            del node['beta']

        MatMul.update_node_stat(node, {
            'transpose_a': node.has_and_set('transpose_a'),
            'transpose_b': node.has_and_set('transpose_b'),
        })
