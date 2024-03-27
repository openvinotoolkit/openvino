# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import float32_array
from openvino.tools.mo.middle.MakeKaldiConstReshapable import create_const_with_batch_from_input
from openvino.tools.mo.ops.MatMul import FullyConnected
from openvino.tools.mo.ops.activation_ops import Tanh, Sigmoid
from openvino.tools.mo.ops.elementwise import Add, Mul
from openvino.tools.mo.ops.split import Split
from openvino.tools.mo.front.caffe.extractors.utils import input_as_const
from openvino.tools.mo.front.common.replacement import FrontReplacementOp
from openvino.tools.mo.front.tf.graph_utils import create_op_with_const_inputs
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.assign import Assign
from openvino.tools.mo.ops.clamp import Clamp
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.read_value import ReadValue
from openvino.tools.mo.ops.result import Result
from openvino.tools.mo.ops.scale_shift import ScaleShiftOp


def unique_id(prefix: str = 'id') -> str:
    """
    Generates a unique id
    The optional string prefix can be specified.
    """
    index = len(unique_id.names)
    name = prefix
    while name in unique_id.names:
        name = '{}_{}'.format(prefix, index)
        index += 1
    unique_id.names.append(name)
    return name


unique_id.names = []


class ReplaceLSTMNodePattern(FrontReplacementOp):
    op = "LSTMCell"
    enabled = True

    def run_after(self):
        from openvino.tools.mo.front.restore_ports import RestorePorts
        return [RestorePorts]

    def run_before(self):
        # current pass should be rewritten to use MatMul ops only (No FullyConnected ops should be created here)
        from openvino.tools.mo.front.MatMul_normalizer import FullyConnectedDecomposer
        from openvino.tools.mo.front.MoveEmbeddedInputsToInputs import MoveEmbeddedInputsToInputs
        return [FullyConnectedDecomposer,
                MoveEmbeddedInputsToInputs]

    def pattern(self):
        return dict(
            nodes=[
                ('op', dict(op=self.__class__.op, format='kaldi'))],
            edges=[]
        )

    def replace_op(self, graph: Graph, node: Node):
        input_out_port = node.in_port(0).get_source()

        memory_pair_input = unique_id('id')
        memory_pair_output = unique_id('id')

        # Input -> FullyConnected
        fc_layer_after_input_attrs = {'name': 'input_fullyconnected',
                                      'out-size': node.gifo_x_weights_shape[0],
                                      'transpose_weights': True,
                                      'bias_term': True,
                                      }

        fc_layer_after_input = FullyConnected(graph, fc_layer_after_input_attrs).create_node()
        fc_layer_after_input.in_port(0).connect(input_out_port)
        input_as_const(fc_layer_after_input, fc_layer_after_input_attrs, 1, 'weights', node.gifo_x_weights)
        input_as_const(fc_layer_after_input, fc_layer_after_input_attrs, 2, 'biases', node.gifo_biases)

        init_value_prev_lstm_output = create_const_with_batch_from_input(input_out_port,
                                                                         node.gifo_r_weights_shape[1])
        prev_lstm_output = ReadValue(graph, {'name': 'prev_memory_output',
                                             'variable_id': memory_pair_input,
                                             'variable_shape': None,
                                             'variable_type': None
                                             }).create_node()
        prev_lstm_output.in_port(0).connect(init_value_prev_lstm_output.out_port(0))

        # *Memory(output) -> FullyConnected
        fc_layer_from_prev_state_attrs = {'name': 'prev_memory_output_fullyconnected',
                                          'out-size': node.gifo_r_weights_shape[0],
                                          'transpose_weights': True,
                                          'bias_term': False,
                                          }

        fc_layer_from_prev_state = FullyConnected(graph, fc_layer_from_prev_state_attrs).create_node()
        fc_layer_from_prev_state.in_port(0).connect(prev_lstm_output.out_port(0))
        input_as_const(fc_layer_from_prev_state, fc_layer_from_prev_state_attrs, 1, 'weights', node.gifo_r_weights)

        # Memory -> FullyConnected  \
        #                           *Eltwise(sum)
        # Input -> FullyConnected   /
        join_input_prev_state_sum = Add(graph, {'name': 'join_input_eltwise'}).create_node()
        join_input_prev_state_sum.in_port(0).connect(fc_layer_from_prev_state.out_port(0))
        join_input_prev_state_sum.in_port(1).connect(fc_layer_after_input.out_port(0))

        # *Eltwise(sum) -> Split
        # it is split into 4 nodes: Act, Eltw*3
        # the following order is mandatory
        #       ___Tanh
        #      /
        # Split ---(2)Eltwise(sum)
        #     |\
        #     | \__(3)Eltwise(sum)
        #     |____(4)Eltwise(sum)
        split_joined_input_axis = Const(graph, {'value': np.int64(1)}).create_node()
        split_joined_input = Split(graph, {'name': 'join_input_split',
                                           'num_splits': 4, 'out_ports_count': 4}).create_node()
        split_joined_input.in_port(0).connect(join_input_prev_state_sum.out_port(0))
        split_joined_input.in_port(1).connect(split_joined_input_axis.out_port(0))

        init_value_prev_lstm_state = create_const_with_batch_from_input(split_joined_input.out_port(0),
                                                                        node.input_gate_weights.shape[0])
        prev_lstm_state = ReadValue(graph, {'name': 'prev_memory_state',
                                            'variable_id': memory_pair_output,
                                            'variable_shape': None,
                                            'variable_type': None
                                            }).create_node()
        prev_lstm_state.in_port(0).connect(init_value_prev_lstm_state.out_port(0))

        # *Memory(state) -> *ScaleShift(input)
        state_input_scaleshift_attrs = {'name': 'input_scaleshift',
                                        'bias_term': False
                                        }
        state_input_scaleshift = ScaleShiftOp(graph, state_input_scaleshift_attrs).create_node()
        state_input_scaleshift.in_port(0).connect(prev_lstm_state.out_port(0))
        input_as_const(state_input_scaleshift, state_input_scaleshift_attrs, 1, 'weights', node.input_gate_weights)

        # *Memory(state) -> *ScaleShift(forget)
        state_forget_scaleshift_attrs = {'name': 'forget_scaleshift',
                                         'bias_term': False
                                         }
        state_forget_scaleshift = ScaleShiftOp(graph, state_forget_scaleshift_attrs).create_node()
        state_forget_scaleshift.in_port(0).connect(prev_lstm_state.out_port(0))
        input_as_const(state_forget_scaleshift, state_forget_scaleshift_attrs, 1, 'weights', node.forget_gate_weights)

        # Split                                 \
        #                                       (2)Eltwise(sum)
        # Memory(state) -> *ScaleShift(input)  /
        join_prev_lstm_input_joined_input_sum = Add(graph, {'name': 'join_prev_lstm_input_joined_input_eltwise'
                                                            }).create_node()
        join_prev_lstm_input_joined_input_sum.in_port(0).connect(split_joined_input.out_port(1))
        join_prev_lstm_input_joined_input_sum.in_port(1).connect(state_input_scaleshift.out_port(0))
        # Split                                 \
        #                                       (3)Eltwise(sum)
        # Memory(state) -> *ScaleShift(forget)  /
        join_prev_lstm_input_joined_forget_sum = Add(graph, {'name': 'join_prev_lstm_input_joined_forget_sum',
                                                             }).create_node()
        join_prev_lstm_input_joined_forget_sum.in_port(0).connect(split_joined_input.out_port(2))
        join_prev_lstm_input_joined_forget_sum.in_port(1).connect(state_forget_scaleshift.out_port(0))

        # Split -> Tanh
        remember_tahn = Tanh(graph, {'name': 'remember_tahnv'}).create_node()
        remember_tahn.in_port(0).connect(split_joined_input.out_port(0))

        # Split -> (2)Eltwise(sum) -> *Sigmoid
        remember_sigmoid = Sigmoid(graph, {'name': 'remember_sigmoid'}).create_node()
        remember_sigmoid.in_port(0).connect(join_prev_lstm_input_joined_input_sum.out_port(0))

        # Split -> (3)Eltwise(sum) -> **Sigmoid
        forget_sigmoid = Sigmoid(graph, {'name': 'forget_sigmoid'}).create_node()
        forget_sigmoid.in_port(0).connect(join_prev_lstm_input_joined_forget_sum.out_port(0))

        # *Memory(state)                        \
        #                                       (6)Eltwise(mul)
        # Split -> (3)Eltwise(sum) -> **Sigmoid /
        join_forget_prev_state_mul = Mul(graph, {'name': 'join_forget_prev_state_mul'}).create_node()
        join_forget_prev_state_mul.in_port(0).connect(forget_sigmoid.out_port(0))
        join_forget_prev_state_mul.in_port(1).connect(prev_lstm_state.out_port(0))

        # Split -> Tahn                         \
        #                                       (5)Eltwise(mul)
        # Split -> (2)Eltwise(sum) -> *Sigmoid   /
        join_remember_candidates_mul = Mul(graph, {'name': 'join_remember_candidates_mul'}).create_node()
        join_remember_candidates_mul.in_port(0).connect(remember_tahn.out_port(0))
        join_remember_candidates_mul.in_port(1).connect(remember_sigmoid.out_port(0))

        # (5)Eltwise(mul)  \
        #               (7)Eltwise(sum)
        # (6)Eltwise(mul)   /
        join_forget_remember_sum = Add(graph, {'name': 'join_forget_remember_sum'}).create_node()
        join_forget_remember_sum.in_port(0).connect(join_forget_prev_state_mul.out_port(0))
        join_forget_remember_sum.in_port(1).connect(join_remember_candidates_mul.out_port(0))

        # (7)Eltwise(sum) -> Clamp
        join_forget_clamp = create_op_with_const_inputs(graph, Clamp, {1: float32_array(-node.clip_value),
                                                                       2: float32_array(node.clip_value)},
                                                        {'name': 'join_forget_clamp'},
                                                        join_forget_remember_sum)
        #
        # Clamp -> (2)Memory(state)
        next_lstm_state = Assign(graph, {'name': 'next_lstm_state',
                                         'variable_id': memory_pair_output}).create_node()
        next_lstm_state.in_port(0).connect(join_forget_clamp.out_port(0))

        res_node = Result(graph, {'name': 'next_lstm_state_out'}).create_node()
        res_node.in_port(0).connect(next_lstm_state.out_port(0))

        # Clamp -> (2)Tahn
        state_filtered_tahn = Tanh(graph, {'name': 'state_filtered_tahn'}).create_node()
        state_filtered_tahn.in_port(0).connect(join_forget_clamp.out_port(0))

        # Clamp -> (2)ScaleShift
        clamp_scaleshift_attrs = {'name': 'clamp_scaleshift',
                                  'bias_term': False}
        clamp_scaleshift = ScaleShiftOp(graph, clamp_scaleshift_attrs).create_node()
        clamp_scaleshift.in_port(0).connect(join_forget_clamp.out_port(0))
        input_as_const(clamp_scaleshift, clamp_scaleshift_attrs, 1, 'weights', node.output_gate_weights)

        # Split                 \
        #                       (4)Eltwise(sum)
        # Clamp -> (2)ScaleShift /
        join_next_lstm_input_joined_input_sum = Add(graph, {'name': 'join_next_lstm_input_joined_input_sum',
                                                            }).create_node()
        join_next_lstm_input_joined_input_sum.in_port(0).connect(split_joined_input.out_port(3))
        join_next_lstm_input_joined_input_sum.in_port(1).connect(clamp_scaleshift.out_port(0))

        # (4)Eltwise(sum) -> (3)Sigmoid
        output_sigmoid = Sigmoid(graph, {'name': 'output_sigmoid'}).create_node()
        output_sigmoid.in_port(0).connect(join_next_lstm_input_joined_input_sum.out_port(0))

        # (4)Eltwise(sum) -> (3)Sigmoid         \
        #                                       (5)Eltwise(mul)
        # Clamp -> (2)Tahn                      /
        joined_output_mul = Mul(graph, {'name': 'joined_output_mul'}).create_node()
        joined_output_mul.in_port(0).connect(state_filtered_tahn.out_port(0))
        joined_output_mul.in_port(1).connect(output_sigmoid.out_port(0))

        # (5)Eltwise(mul) -> (3)FullyConnected
        fc_output_attrs = {'name': 'FullyConnected',
                           'out-size': node.projection_weights_shape[0],
                           'transpose_weights': True,
                           'bias_term': False}
        fc_output = FullyConnected(graph, fc_output_attrs).create_node()
        fc_output.in_port(0).connect(joined_output_mul.out_port(0))
        input_as_const(fc_output, fc_output_attrs, 1, 'weights', node.projection_weights)

        #                   / (2)Memory(output)
        # (3)FullyConnected
        #                   \ Output (any next node) (edge created automatically after replacement)
        next_lstm_output = Assign(graph, {'name': 'next_lstm_output',
                                          'variable_id': memory_pair_input}).create_node()
        next_lstm_output.in_port(0).connect(fc_output.out_port(0))

        res_node_lstm_output = Result(graph, {'name': 'next_lstm_output_out'}).create_node()
        res_node_lstm_output.in_port(0).connect(next_lstm_output.out_port(0))

        return [fc_output.id]
