"""
 Copyright (C) 2018-2020 Intel Corporation

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
from extensions.ops.elementwise import Mul, Pow
from extensions.ops.split import VariadicSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementPattern
from mo.front.tf.graph_utils import create_op_with_const_inputs, create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.shape import Shape


class ReplaceConvolutionReshape(FrontReplacementPattern):
    """
       This pass adds Reshapes around a Convolution layer for reshaping from NH to NCHW
       For example:
           Let's suppose we have next graph:

           Prev_Layer [N, H] -> Convolution [N, C, H, W] -> Next_Layer [N, H]

           In this case Convolution takes only [N, H] from input tensor in 3rd dim
           So this pass will convert this graph to the next one:

           Prev_Layer [N, H] -> Reshape -> Convolution [N, C, H, W] -> Reshape -> Next_Layer [N, H]

   """
    enabled = True
    graph_condition = [lambda graph: graph.graph['fw'] == "kaldi"]

    def run_before(self):
        from extensions.front.kaldi.add_permute_after_convolution import ReplaceConvolutionTranspose
        return [ReplaceConvolutionTranspose]

    def pattern(self):
        return dict(nodes=[('conv', dict(op='Convolution'))],
                    edges=[])

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['conv']
        node_name = node.soft_get('name', node.id)

        # create Reshape before convolution
        # shape = [in_shape[0], in_shape[1]/patch_stride, 1, patch_stride]
        shape = Shape(graph, {'name': node_name + '/Shape'}).create_node()
        shape.in_port(0).connect(node.in_port(0).get_source())

        split = create_op_with_const_inputs(graph, VariadicSplit, {1: int64_array(0), 2: int64_array([1, -1])},
                                            {'name': shape.name + '/split_batch', 'out_ports_count': 2}, shape)

        pow_node = create_op_node_with_second_input(graph, Pow, int64_array([-1]), {'name': node_name + '/patch_stride/inverse'})
        conv_patch_stride = Const(graph, {'value': int64_array([node.patch_stride]),
                                          'name': node_name + '/patch_stride/'}).create_node()
        pow_node.in_port(0).connect(conv_patch_stride.out_port(0))

        mul = Mul(graph, {'name': node_name + '/mul_inverse_stride_h'}).create_node()
        mul.in_port(0).connect(split.out_port(1))
        mul.in_port(1).connect(pow_node.out_port(0))

        concat = create_op_with_const_inputs(graph, Concat, {2: int64_array([1])},
                                             {'name': node_name + '/concat_all_dims', 'in_ports_count': 4, 'axis': 0})

        concat.in_port(0).connect(split.out_port(0))
        concat.in_port(1).connect(mul.out_port(0))
        concat.in_port(3).connect(conv_patch_stride.out_port(0))

        reshape_in = Reshape(graph, {'name': node_name + '/reshape_in'}).create_node()
        reshape_in.in_port(1).connect(concat.out_port(0))

        # create Reshape after Convolution
        reshape_out = create_op_node_with_second_input(graph, Reshape, int64_array([0, -1]),
                                                       {'name': node_name + '/reshape_out'})

        # connect input_reshape_node
        source = node.in_port(0).get_source()
        node.in_port(0).get_connection().set_source(reshape_in.out_port(0))
        reshape_in.in_port(0).connect(source)
        # connect output_reshape_node
        node.out_port(0).get_connection().set_source(reshape_out.out_port(0))
        node.out_port(0).connect(reshape_out.in_port(0))
