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
from extensions.ops.activation_ops import Sigmoid, Tanh
from mo.front.caffe.extractors.utils import embed_input
from mo.front.common.replacement import FrontReplacementOp
from mo.graph.graph import Node, Graph
from mo.ops.concat import Concat
from mo.ops.eltwise import Eltwise
from mo.ops.scale_shift import ScaleShiftOp
from mo.ops.split import Split


class ReplaceLstmNonLinearityPattern(FrontReplacementOp):
    op = "LstmNonLinearity"
    enabled = True

    def run_after(self):
        from extensions.front.restore_ports import RestorePorts
        return [RestorePorts]

    def replace_op(self, graph: Graph, node: Node):
        # split input to (i_part, f_part, c_part, o_part, ct_1)
        split_node = Split(graph, {'name': graph.unique_id(prefix='Split_lstm_input_'),
                                   'num_split': 5}).create_node()
        node.in_port(0).get_connection().set_destination(split_node.in_port(0))
        for i in range(5):
            split_node.add_output_port(i)

        # i_t = Sigmoid(i_part + w_ic*ct_1)
        i_scale_attrs = {'name': graph.unique_id(prefix='i_scaleshift'),
                         'bias_term': False}
        embed_input(i_scale_attrs, 1, 'weights', node.i_weights)
        i_scale = ScaleShiftOp(graph, i_scale_attrs).create_node()
        split_node.out_port(4).connect(i_scale.in_port(0))

        sum_i_c = Eltwise(graph, {'name': graph.unique_id(prefix='sum_i_c_'), 'operation': 'sum'}).create_node()
        split_node.out_port(0).connect(sum_i_c.in_port(0))
        i_scale.out_port(0).connect(sum_i_c.in_port(1))

        i_sigmoid = Sigmoid(graph, {'name': 'i_sigmoid'}).create_node()
        sum_i_c.out_port(0).connect(i_sigmoid.in_port(0))

        # f_t = Sigmoid(f_part + w_fc*ct_1)
        f_scale_attrs = {'name': graph.unique_id(prefix='f_scaleshift'),
                         'bias_term': False}
        embed_input(f_scale_attrs, 1, 'weights', node.f_weights)
        f_scale = ScaleShiftOp(graph, f_scale_attrs).create_node()
        split_node.out_port(4).connect(f_scale.in_port(0))

        sum_f_c = Eltwise(graph, {'name': graph.unique_id(prefix='sum_f_c_'), 'operation': 'sum'}).create_node()
        split_node.out_port(1).connect(sum_f_c.in_port(0))
        f_scale.out_port(0).connect(sum_f_c.in_port(1))

        f_sigmoid = Sigmoid(graph, {'name': 'f_sigmoid'}).create_node()
        sum_f_c.out_port(0).connect(f_sigmoid.in_port(0))

        # c_t = f_t*ct_1 + i_t * tanh(c_part)
        c_tanh = Tanh(graph, {'name': 'c_tanh'}).create_node()
        split_node.out_port(2).connect(c_tanh.in_port(0))

        prod_i_c_tanh = Eltwise(graph, {'name': graph.unique_id(prefix='prod_i_c_tanh_'),
                                        'operation': 'mul'}).create_node()
        i_sigmoid.out_port(0).connect(prod_i_c_tanh.in_port(0))
        c_tanh.out_port(0).connect(prod_i_c_tanh.in_port(1))

        prod_f_ct_1 = Eltwise(graph, {'name': graph.unique_id(prefix='prod_f_ct_1_'),
                                      'operation': 'mul'}).create_node()
        f_sigmoid.out_port(0).connect(prod_f_ct_1.in_port(0))
        split_node.out_port(4).connect(prod_f_ct_1.in_port(1))

        sum_f_i = Eltwise(graph, {'name': graph.unique_id(prefix='sum_f_i_'), 'operation': 'sum'}).create_node()
        prod_f_ct_1.out_port(0).connect(sum_f_i.in_port(0))
        prod_i_c_tanh.out_port(0).connect(sum_f_i.in_port(1))

        #  o_t = Sigmoid(o_part + w_oc*c_t)
        o_scale_attrs = {'name': graph.unique_id(prefix='o_scaleshift'),
                         'bias_term': False}
        embed_input(o_scale_attrs, 1, 'weights', node.o_weights)
        o_scale = ScaleShiftOp(graph, o_scale_attrs).create_node()
        sum_f_i.out_port(0).connect(o_scale.in_port(0))

        sum_o_c = Eltwise(graph, {'name': graph.unique_id(prefix='sum_o_c_'), 'operation': 'sum'}).create_node()
        split_node.out_port(3).connect(sum_o_c.in_port(0))
        o_scale.out_port(0).connect(sum_o_c.in_port(1))

        o_sigmoid = Sigmoid(graph, {'name': 'o_sigmoid'}).create_node()
        sum_o_c.out_port(0).connect(o_sigmoid.in_port(0))

        # m_t = o_t * Tanh(c_t)
        c_t_tanh = Tanh(graph, {'name': 'c_t_tanh'}).create_node()
        sum_f_i.out_port(0).connect(c_t_tanh.in_port(0))

        prod_o_c_t_tanh = Eltwise(graph, {'name': graph.unique_id(prefix='prod_o_c_t_tanh_'),
                                          'operation': 'mul'}).create_node()
        o_sigmoid.out_port(0).connect(prod_o_c_t_tanh.in_port(0))
        c_t_tanh.out_port(0).connect(prod_o_c_t_tanh.in_port(1))

        # add concat to create 1 output
        concat = Concat(graph, {'name': graph.unique_id(prefix='Concat_c_m')}).create_node()
        concat.add_sequence_of_ports('in', range(2))
        sum_f_i.out_port(0).connect(concat.in_port(0))
        prod_o_c_t_tanh.out_port(0).connect(concat.in_port(1))

        return [concat.id]
