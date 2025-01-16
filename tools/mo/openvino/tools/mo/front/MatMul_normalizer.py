# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

from openvino.tools.mo.ops.MatMul import MatMul
from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.common.replacement import FrontReplacementSubgraph
from openvino.tools.mo.front.subgraph_matcher import SubgraphMatch
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.ops.reshape import Reshape


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
            node_name = node.name + '/WithoutBiases'
            bias_node_name = node.name
            rename_nodes([(node, node_name), (bias_node, bias_node_name)])
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

        # input normalization for 4D Caffe FullyConnected
        if graph.graph['fw'] == 'caffe':
            node.insert_op_on_input_port(in_port_idx=0, new_op_class=Reshape,
                                         new_op_attrs={'name': name + '/flatten_fc_input', 'special_zero': True},
                                         value=int64_array([0, -1]))

        MatMul.update_node_stat(node, {})


class GemmDecomposer(FrontReplacementSubgraph):
    """
    Decomposes Gemm operation:
        1. Biases are added separately with the help of Add node
        2. Multiplication by `alpha` and `beta` values are separated to Mul operations
        3. Gemm operation itself is converted to MatMul
    """
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(op='Gemm'):
            name = node.soft_get('name', node.id)
            node_output_port = node.out_port(0)
            if node.has_valid('alpha') and not math.isclose(node.alpha, 1):
                mul_alpha = create_op_with_const_inputs(graph, Mul, {1: mo_array(node.alpha)},
                                                       {'name': name + '/Alpha', 'can_be_scaleshift': False})
                node_output_port.get_connection().insert_node(mul_alpha)
                node_output_port = mul_alpha.out_port(0)
                del node['alpha']

            if node.is_in_port_connected(2):
                # biases normalization
                bias_node = Add(graph, {'name': name + '/Bias_', 'can_be_scaleshift': False}).create_node()
                without_biases_node_name = name + '/WithoutBiases'
                rename_nodes([(node, without_biases_node_name), (bias_node, name)])
                node_output_port.get_connection().set_source(bias_node.out_port(0))
                node.in_port(2).get_connection().set_destination(bias_node.in_port(1))
                node_output_port.connect(bias_node.in_port(0))
                if node.has_valid('beta') and not math.isclose(node.beta, 1):
                    bias_node.insert_op_on_input_port(in_port_idx=1, new_op_class=Mul, value=mo_array(node.beta),
                                                      new_op_attrs={'name': name + '/Beta',
                                                                    'can_be_scaleshift': False})
                    del node['beta']

            MatMul.update_node_stat(node, {
                'transpose_a': node.has_and_set('transpose_a'),
                'transpose_b': node.has_and_set('transpose_b'),
            })
