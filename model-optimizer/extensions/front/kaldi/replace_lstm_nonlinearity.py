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
import numpy as np

from extensions.ops.activation_ops import Sigmoid, Tanh
from extensions.ops.elementwise import Add, Mul
from extensions.ops.split import Split
from mo.front.caffe.extractors.utils import input_as_const
from mo.front.common.replacement import FrontReplacementOp
from mo.front.tf.graph_utils import create_op_with_const_inputs
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.scale_shift import ScaleShiftOp


class ReplaceLstmNonLinearityPattern(FrontReplacementOp):
    op = "LstmNonLinearity"
    enabled = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        from extensions.front.MatMul_normalizer import FullyConnectedDecomposer
        return [FullyConnectedDecomposer]

    def replace_op(self, graph: Graph, node: Node):
        # split input to (i_part, f_part, c_part, o_part, ct_1)
        split_node = create_op_with_const_inputs(graph, Split, {1: np.int64(1)}, {'name': 'Split_lstm_input_',
                                   'num_splits': 5})
        node.in_port(0).get_connection().set_destination(split_node.in_port(0))

        # i_t = Sigmoid(i_part + w_ic*ct_1)
        i_scale_attrs = {'name': 'i_scaleshift',
                         'bias_term': False}
        i_scale = ScaleShiftOp(graph, i_scale_attrs).create_node()
        input_as_const(i_scale, i_scale_attrs, 1, 'weights', node.i_weights)
        split_node.out_port(4).connect(i_scale.in_port(0))

        sum_i_c = Add(graph, {'name': 'sum_i_c_'}).create_node()
        split_node.out_port(0).connect(sum_i_c.in_port(0))
        i_scale.out_port(0).connect(sum_i_c.in_port(1))

        i_sigmoid = Sigmoid(graph, {'name': 'i_sigmoid'}).create_node()
        sum_i_c.out_port(0).connect(i_sigmoid.in_port(0))

        # f_t = Sigmoid(f_part + w_fc*ct_1)
        f_scale_attrs = {'name': 'f_scaleshift',
                         'bias_term': False}
        f_scale = ScaleShiftOp(graph, f_scale_attrs).create_node()
        input_as_const(f_scale, f_scale_attrs, 1, 'weights', node.f_weights)
        split_node.out_port(4).connect(f_scale.in_port(0))

        sum_f_c = Add(graph, {'name': 'sum_f_c_'}).create_node()
        split_node.out_port(1).connect(sum_f_c.in_port(0))
        f_scale.out_port(0).connect(sum_f_c.in_port(1))

        f_sigmoid = Sigmoid(graph, {'name': 'f_sigmoid'}).create_node()
        sum_f_c.out_port(0).connect(f_sigmoid.in_port(0))

        # c_t = f_t*ct_1 + i_t * tanh(c_part)
        c_tanh = Tanh(graph, {'name': 'c_tanh'}).create_node()
        split_node.out_port(2).connect(c_tanh.in_port(0))

        prod_i_c_tanh = Mul(graph, {'name': 'prod_i_c_tanh_'}).create_node()
        i_sigmoid.out_port(0).connect(prod_i_c_tanh.in_port(0))
        c_tanh.out_port(0).connect(prod_i_c_tanh.in_port(1))

        prod_f_ct_1 = Mul(graph, {'name': 'prod_f_ct_1_'}).create_node()
        f_sigmoid.out_port(0).connect(prod_f_ct_1.in_port(0))
        split_node.out_port(4).connect(prod_f_ct_1.in_port(1))

        sum_f_i = Add(graph, {'name': 'sum_f_i_'}).create_node()
        prod_f_ct_1.out_port(0).connect(sum_f_i.in_port(0))
        prod_i_c_tanh.out_port(0).connect(sum_f_i.in_port(1))

        #  o_t = Sigmoid(o_part + w_oc*c_t)
        o_scale_attrs = {'name': 'o_scaleshift',
                         'bias_term': False}
        o_scale = ScaleShiftOp(graph, o_scale_attrs).create_node()
        input_as_const(o_scale, o_scale_attrs, 1, 'weights', node.o_weights)
        sum_f_i.out_port(0).connect(o_scale.in_port(0))

        sum_o_c = Add(graph, {'name': 'sum_o_c_'}).create_node()
        split_node.out_port(3).connect(sum_o_c.in_port(0))
        o_scale.out_port(0).connect(sum_o_c.in_port(1))

        o_sigmoid = Sigmoid(graph, {'name': 'o_sigmoid'}).create_node()
        sum_o_c.out_port(0).connect(o_sigmoid.in_port(0))

        # m_t = o_t * Tanh(c_t)
        c_t_tanh = Tanh(graph, {'name': 'c_t_tanh'}).create_node()
        sum_f_i.out_port(0).connect(c_t_tanh.in_port(0))

        prod_o_c_t_tanh = Mul(graph, {'name': 'prod_o_c_t_tanh_'}).create_node()
        o_sigmoid.out_port(0).connect(prod_o_c_t_tanh.in_port(0))
        c_t_tanh.out_port(0).connect(prod_o_c_t_tanh.in_port(1))

        # add concat to create 1 output
        concat = Concat(graph, {'name': 'Concat_c_m'}).create_node()
        concat.add_sequence_of_ports('in', range(2))
        sum_f_i.out_port(0).connect(concat.in_port(0))
        prod_o_c_t_tanh.out_port(0).connect(concat.in_port(1))

        return [concat.id]
