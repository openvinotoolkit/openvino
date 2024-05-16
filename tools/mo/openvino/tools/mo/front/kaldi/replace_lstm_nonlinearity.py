# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.activation_ops import Sigmoid, Tanh
from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.ops.split import Split, AttributedVariadicSplit
from openvino.tools.mo.front.caffe.extractors.utils import input_as_const
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.concat import Concat
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp


class ReplaceLstmNonLinearityPattern(FrontReplacementOp):
    op = "LstmNonLinearity"
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        from openvino.tools.mo.front.MatMul_normalizer import FullyConnectedDecomposer
        return [FullyConnectedDecomposer]

    def replace_op(self, graph: Graph, node: Node):
        node_name = node.soft_get('name', node.id)
        # check if we have dropout
        input_port = node.in_port(0)
        if node.has_and_set('use_dropout'):
            split_dropout = AttributedVariadicSplit(graph,
                                                    {'name': node_name + '/split_dropout',
                                                     'size_splits': int64_array([-1, 1, 1, 1]),
                                                     'axis': int64_array(1)}).create_node()
            input_port.get_connection().set_destination(split_dropout.in_port(0))
            input_port = split_dropout.out_port(0)
            i_drop_scale = split_dropout.out_port(1)
            f_drop_scale = split_dropout.out_port(2)
            o_drop_scale = split_dropout.out_port(3)

        # split input to (i_part, f_part, c_part, o_part, ct_1)
        split_node = create_op_with_const_inputs(graph, Split, {1: np.int64(1)},
                                                 {'name': node_name + '/split_lstm_input',
                                                  'num_splits': 5})
        input_port.get_connection().set_destination(split_node.in_port(0))

        i_part = split_node.out_port(0)
        f_part = split_node.out_port(1)
        c_part = split_node.out_port(2)
        o_part = split_node.out_port(3)
        ct_1 = split_node.out_port(4)

        # i_t = Sigmoid(i_part + w_ic*ct_1)
        i_scale_attrs = {'name': node_name + '/i_scaleshift',
                         'bias_term': False}
        i_scale = ScaleShiftOp(graph, i_scale_attrs).create_node()
        input_as_const(i_scale, i_scale_attrs, 1, 'weights', node.i_weights)
        ct_1.connect(i_scale.in_port(0))

        sum_i_c = Add(graph, {'name': node_name + '/sum_i_c_'}).create_node()
        i_part.connect(sum_i_c.in_port(0))
        i_scale.out_port(0).connect(sum_i_c.in_port(1))

        i_sigmoid = Sigmoid(graph, {'name': node_name + '/i_sigmoid'}).create_node()
        sum_i_c.out_port(0).connect(i_sigmoid.in_port(0))

        if node['use_dropout']:
            mul_dropout_i = Mul(graph, {'name': split_node.soft_get('name', split_node.id) + '/mul_i'}).create_node()
            mul_dropout_i.in_port(0).connect(i_sigmoid.out_port(0))
            mul_dropout_i.in_port(1).connect(i_drop_scale)  # pylint: disable=possibly-used-before-assignment
            i_sigmoid = mul_dropout_i

        # f_t = Sigmoid(f_part + w_fc*ct_1)
        f_scale_attrs = {'name': node_name + '/f_scaleshift',
                         'bias_term': False}
        f_scale = ScaleShiftOp(graph, f_scale_attrs).create_node()
        input_as_const(f_scale, f_scale_attrs, 1, 'weights', node.f_weights)
        ct_1.connect(f_scale.in_port(0))

        sum_f_c = Add(graph, {'name': node_name + '/sum_f_c_'}).create_node()
        f_part.connect(sum_f_c.in_port(0))
        f_scale.out_port(0).connect(sum_f_c.in_port(1))

        f_sigmoid = Sigmoid(graph, {'name': node_name + '/f_sigmoid'}).create_node()
        sum_f_c.out_port(0).connect(f_sigmoid.in_port(0))

        if node['use_dropout']:
            mul_dropout_f = Mul(graph, {'name': split_node.soft_get('name', split_node.id) + '/mul_f'}).create_node()
            mul_dropout_f.in_port(0).connect(f_sigmoid.out_port(0))
            mul_dropout_f.in_port(1).connect(f_drop_scale)  # pylint: disable=possibly-used-before-assignment
            f_sigmoid = mul_dropout_f

        # c_t = f_t*ct_1 + i_t * tanh(c_part)
        c_tanh = Tanh(graph, {'name': node_name + '/c_tanh'}).create_node()
        c_part.connect(c_tanh.in_port(0))

        prod_i_c_tanh = Mul(graph, {'name': node_name + '/prod_i_c_tanh_'}).create_node()
        i_sigmoid.out_port(0).connect(prod_i_c_tanh.in_port(0))
        c_tanh.out_port(0).connect(prod_i_c_tanh.in_port(1))

        prod_f_ct_1 = Mul(graph, {'name': node_name + '/prod_f_ct_1_'}).create_node()
        f_sigmoid.out_port(0).connect(prod_f_ct_1.in_port(0))
        ct_1.connect(prod_f_ct_1.in_port(1))

        sum_f_i = Add(graph, {'name': node_name + '/sum_f_i_'}).create_node()
        prod_f_ct_1.out_port(0).connect(sum_f_i.in_port(0))
        prod_i_c_tanh.out_port(0).connect(sum_f_i.in_port(1))

        #  o_t = Sigmoid(o_part + w_oc*c_t)
        o_scale_attrs = {'name': node_name + '/o_scaleshift',
                         'bias_term': False}
        o_scale = ScaleShiftOp(graph, o_scale_attrs).create_node()
        input_as_const(o_scale, o_scale_attrs, 1, 'weights', node.o_weights)
        sum_f_i.out_port(0).connect(o_scale.in_port(0))

        sum_o_c = Add(graph, {'name': node_name + '/sum_o_c_'}).create_node()
        o_part.connect(sum_o_c.in_port(0))
        o_scale.out_port(0).connect(sum_o_c.in_port(1))

        o_sigmoid = Sigmoid(graph, {'name': node_name + '/o_sigmoid'}).create_node()
        sum_o_c.out_port(0).connect(o_sigmoid.in_port(0))

        if node['use_dropout']:
            mul_dropout_o = Mul(graph, {'name': split_node.soft_get('name', split_node.id) + '/mul_o'}).create_node()
            mul_dropout_o.in_port(0).connect(o_sigmoid.out_port(0))
            mul_dropout_o.in_port(1).connect(o_drop_scale)  # pylint: disable=possibly-used-before-assignment
            o_sigmoid = mul_dropout_o

        # m_t = o_t * Tanh(c_t)
        c_t_tanh = Tanh(graph, {'name': node_name + '/c_t_tanh'}).create_node()
        sum_f_i.out_port(0).connect(c_t_tanh.in_port(0))

        prod_o_c_t_tanh = Mul(graph, {'name': node_name + '/prod_o_c_t_tanh_'}).create_node()
        o_sigmoid.out_port(0).connect(prod_o_c_t_tanh.in_port(0))
        c_t_tanh.out_port(0).connect(prod_o_c_t_tanh.in_port(1))

        # add concat to create 1 output
        concat = Concat(graph, {'name': node_name + '/concat_c_m'}).create_node()
        concat.add_sequence_of_ports('in', range(2))
        sum_f_i.out_port(0).connect(concat.in_port(0))
        prod_o_c_t_tanh.out_port(0).connect(concat.in_port(1))

        return [concat.id]
