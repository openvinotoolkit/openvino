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
import unittest

import numpy as np

from extensions.front.kaldi.replace_lstm_nonlinearity import ReplaceLstmNonLinearityPattern
from mo.graph.graph import Node
from mo.utils.ir_engine.compare_graphs import compare_graphs
from mo.utils.unittest.graph import build_graph


class ReplaceLstmNonlinearityTests(unittest.TestCase):
    # i_t = Sigmoid(i_part + w_ic*ct_1)
    # f_t = Sigmoid(f_part + w_fc*ct_1)
    # c_t = f_t*ct_1 + i_t * tanh(c_part)
    # o_t = Sigmoid(o_part + w_oc*c_t)
    # m_t = o_t * Tanh(c_t)
    nodes_attributes = {
        'in': {'kind': 'op', 'op': 'Parameter'},
        'i_part': {'kind': 'op', 'op': 'Parameter'},
        'f_part': {'kind': 'op', 'op': 'Parameter'},
        'c_part': {'kind': 'op', 'op': 'Parameter'},
        'o_part': {'kind': 'op', 'op': 'Parameter'},
        'split': {'kind': 'op', 'op': 'Split'},
        'sigmoid_i': {'kind': 'op', 'op': 'Sigmoid'},
        'sigmoid_f': {'kind': 'op', 'op': 'Sigmoid'},
        'sigmoid_o': {'kind': 'op', 'op': 'Sigmoid'},
        'i_plus_c': {'kind': 'op', 'op': 'Eltwise', 'operation': 'sum'},
        'f_plus_c': {'kind': 'op', 'op': 'Eltwise', 'operation': 'sum'},
        'fc_plus_itanhc': {'kind': 'op', 'op': 'Eltwise', 'operation': 'sum'},
        'o_plus_c': {'kind': 'op', 'op': 'Eltwise', 'operation': 'sum'},
        'scale_i_c': {'kind': 'op', 'op': 'ScaleShift'},
        'scale_f_c': {'kind': 'op', 'op': 'ScaleShift'},
        'scale_o_c': {'kind': 'op', 'op': 'ScaleShift'},
        'f_mul_c': {'kind': 'op', 'op': 'Eltwise', 'operation': 'mul'},
        'i_mul_tanhc': {'kind': 'op', 'op': 'Eltwise', 'operation': 'mul'},
        'o_mul_tanhc': {'kind': 'op', 'op': 'Eltwise', 'operation': 'mul'},
        'tanhcp': {'kind': 'op', 'op': 'Tanh'},
        'tanhc': {'kind': 'op', 'op': 'Tanh'},
        'concat': {'kind': 'op', 'op': 'Concat'},
        'out': {'kind': 'op', 'op': 'Placeholder'},
        'lstm': {'kind': 'op', 'op': 'LstmNonLinearity',
                 'i_weights': np.array([]),
                 'f_weights': np.array([]),
                 'o_weights': np.array([])}
    }

    def test_lstm_nonlinearity(self):
        graph = build_graph({'in': {'kind': 'op', 'op': 'Parameter'},
                             'lstm': {'kind': 'op', 'op': 'LstmNonLinearity',
                                      'i_weights': np.array([]),
                                      'f_weights': np.array([]),
                                      'o_weights': np.array([]),},
                             'out': {'kind': 'op', 'op': 'Placeholder'}},
                            [('in', 'lstm'), ('lstm', 'out')], nodes_with_edges_only=True)
        graph.stage = 'front'
        # split input to (i_part, f_part, c_part, o_part, ct_1)
        ref_graph = build_graph(self.nodes_attributes, [
            ('in', 'split'),
            ('split', 'scale_i_c', {'out': 4}),
            ('scale_i_c', 'i_plus_c'),
            ('split', 'i_plus_c', {'out': 0}),
            ('i_plus_c', 'sigmoid_i'),
            ('split', 'scale_f_c', {'out': 4}),
            ('scale_f_c', 'f_plus_c'),
            ('split', 'f_plus_c', {'out': 1}),
            ('f_plus_c', 'sigmoid_f'),
            ('split', 'tanhcp', {'out': 2}),
            ('tanhcp', 'i_mul_tanhc'),
            ('sigmoid_i', 'i_mul_tanhc'),
            ('sigmoid_f', 'f_mul_c'),
            ('split', 'f_mul_c', {'out': 4}),
            ('f_mul_c', 'fc_plus_itanhc'),
            ('i_mul_tanhc', 'fc_plus_itanhc'),
            ('split', 'scale_o_c', {'out': 4}),
            ('scale_o_c', 'o_plus_c'),
            ('split', 'o_plus_c', {'out': 3}),
            ('o_plus_c', 'sigmoid_o'),
            ('fc_plus_itanhc', 'tanhc'),
            ('sigmoid_o', 'o_mul_tanhc'),
            ('tanhc', 'o_mul_tanhc'),
            ('fc_plus_itanhc', 'concat'),
            ('o_mul_tanhc', 'concat'),
            ('lstm', 'out'),
        ], nodes_with_edges_only=True)
        ReplaceLstmNonLinearityPattern().replace_op(graph, Node(graph, 'lstm'))
        (flag, resp) = compare_graphs(graph, ref_graph, 'out', check_op_attrs=True)
        self.assertTrue(flag, resp)
