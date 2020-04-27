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
from extensions.ops.Log import LogOp
from extensions.ops.ReduceOps import ReduceMax, ReduceSum
from extensions.ops.elementwise import Sub
from extensions.ops.exp import ExpOp
from mo.front.common.partial_infer.utils import int64_array
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Graph, Node, rename_nodes
from mo.ops.const import Const


class LogSoftmaxFrontReplacer(FrontReplacementOp):
    """
    Replace LogSoftmax operation with ReduceMax + Sub + Exp + ReduceSum + Log + Sub.

    More precisely, this transformation implements the following formulas of the calculation of LogSoftmax:

        shifted_data = input_data - ReduceMax(input_data, axis),              (1)
        output = shifted_data - Log(ReduceSum(Exp(shifted_data), axis)).

    These formulas is used to calculate LogSoftmax in implementation of TensorFlow (see
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/softmax_op_functor.h),
    Kaldi (see https://github.com/kaldi-asr/kaldi/blob/master/src/cudamatrix/cu-kernels.cu),
    MxNet (see https://github.com/apache/incubator-mxnet/blob/master/src/operator/nn/softmax-inl.h).

    ONNX implements LogSoftmax according to formulas

        flatten_data = Flatten(input_data, axis),                              (1')
        shifted_data = flatten_data - ReduceMax(flatten_data, 1),
        z = shifted_data - Log(ReduceSum(Exp(shifted_data), 1)),
        output = Reshape(z, input_data.shape)

    (see https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/codegen/mti/math/logsoftmax.cc,
     https://github.com/microsoft/onnxruntime-tvm/blob/master/topi/include/topi/nn/softmax.h)

     Formally speaking, the formula (1) is equivalent to the formula
        output = Log(SoftMax(input_data, axis)) (2)

    But LogSoftMax is calculated according to formula (1) for better numeric stability.
    """
    op = "LogSoftmax"
    enabled = True

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
        assert node.has_valid('axis'), 'The node "{}" does not have mandatory attribute "axis"'.format(node_name)

        # Creating of ReduceMax -> Sub -> Exp block
        first_sub_node = Sub(graph, {'name': node_name + '/Sub_/first_'}).create_node()
        reduce_max_node = create_op_with_const_inputs(graph,
                                                      ReduceMax,
                                                      {1: int64_array([node.axis])},
                                                      op_attrs={'name': node_name + '/ReduceMax_', 'keep_dims': True})
        reduce_max_node.out_port(0).connect(first_sub_node.in_port(1))

        # Creating of Exp -> ReduceSum -> Log block
        exp_node = ExpOp(graph,  {'name': node_name + '/Exp_'}).create_node()
        reduce_sum_node = create_op_with_const_inputs(graph,
                                                      ReduceSum,
                                                      {1: int64_array([node.axis])},
                                                      op_attrs={'name': node_name + '/ReduceSum_', 'keep_dims': True})
        log_node = LogOp(graph, {'name': node_name + '/Log_'}).create_node()

        first_sub_node.out_port(0).connect(exp_node.in_port(0))
        exp_node.out_port(0).connect(reduce_sum_node.in_port(0))
        reduce_sum_node.out_port(0).connect(log_node.in_port(0))

        # Creating of the last Sub node
        second_sub_node = Sub(graph, {}).create_node()
        rename_nodes([(node, node_name + '/delete'), (second_sub_node, node_name)])
        log_node.out_port(0).connect(second_sub_node.in_port(1))
        first_sub_node.out_port(0).connect(second_sub_node.in_port(0))

        # Correcting of input edges
        source = node.in_port(0).get_source()
        first_sub_node.in_port(0).connect(source)
        reduce_max_node.in_port(0).connect(source)

        return [second_sub_node.id]
