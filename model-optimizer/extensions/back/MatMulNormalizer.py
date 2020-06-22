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

from extensions.ops.transpose import Transpose
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.unsqueeze import Unsqueeze


class MatMulConstTransposesExtraction(BackReplacementPattern):
    """
    Resolves transpose_a(b) key from MatMul operation if corresponding input is constant by inserting Transpose,
    that gets const folded while graph clean up execution
    """

    enabled = True
    force_clean_up = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[('matmul', dict(kind='op', op='MatMul'))],
            edges=[]
        )

    @staticmethod
    def insert_transpose(node, in_port_idx):
        graph = node.graph
        name = node.soft_get('name', node.id)

        assert in_port_idx in node.in_ports() and not node.in_port(in_port_idx).disconnected(), \
            'Input port with index {} should be connected for node {}'.format(in_port_idx, name)

        in_port = node.in_port(in_port_idx)
        port_shape = in_port.data.get_shape()
        assert port_shape is not None, \
            'Shape is unknown for input port with index {} for node {}'.format(in_port_idx, name)

        transpose_order = list(range(port_shape.size))
        transpose_order[-1], transpose_order[-2] = transpose_order[-2], transpose_order[-1]

        transpose = Transpose(graph, {'name': name + '/{}_port_transpose'.format(in_port_idx)}).create_node()
        order = Const(graph, {'value': int64_array(transpose_order), 'name': transpose.name + '/order'}).create_node()

        port_source = in_port.get_source()
        in_port.get_connection().set_source(transpose.out_port(0))
        transpose.in_port(0).connect(port_source)
        transpose.in_port(1).connect(order.out_port(0))

        transpose['override_output_shape'] = True

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['matmul']

        if not node.has_and_set('transpose_b'):
            B_shape = node.in_port(1).data.get_shape()
            B_value = node.in_port(1).data.get_value()
            FQ_on_weights = node.in_port(1).get_source().node.has_and_set('stop_value_propagation')
            if (B_value is not None or FQ_on_weights) and B_shape[B_shape != 1].size <= 2:
                MatMulConstTransposesExtraction.insert_transpose(node, 1)
                node['transpose_b'] = True


class PullTransposeThroughFQUp(BackReplacementPattern):
    """
        BEFORE                                      AFTER
                                                        T  T T  T  T
         \ \ | / /                                       \ \ | / /
        FakeQuantize                                    FakeQuantize
            |                                                |
        Transpose                                         next_op
            |
         next_op

        `T` is Transpose for short
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [MatMulConstTransposesExtraction]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('fq', dict(kind='op', type='FakeQuantize')),
                ('data', dict()),
                ('transpose', dict(kind='op', type='Transpose')),
            ],
            edges=[
                ('fq', 'data'),
                ('data', 'transpose'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        fq = match['fq']
        transpose = match['transpose']
        name = fq.soft_get('name', fq.id)

        input_shape = transpose.in_port(0).data.get_shape()

        # detaching transpose from the graph
        transpose.out_port(0).get_connection().set_source(transpose.in_port(0).get_connection().get_source())
        transpose.in_port(0).disconnect()

        for idx, port in fq.in_ports().items():
            transpose_copy = transpose.copy_node({'override_output_shape': True})
            transpose.in_port(1).get_source().connect(transpose_copy.in_port(1))

            start_port = transpose_copy.in_port(0)

            idxs = np.arange(len(input_shape) - len(port.data.get_shape()))
            if idxs.size != 0:
                axis = Const(graph, {'name': name + '/in_{}_unsqueeze_axis'.format(idx),
                                     'value': int64_array(idxs)}).create_node()
                unsqueeze = Unsqueeze(graph, {'name': name + '/in_{}_unsqueeze'.format(idx)}).create_node()
                axis.out_port(0).connect(unsqueeze.in_port(1))
                unsqueeze.out_port(0).connect(transpose_copy.in_port(0))
                start_port = unsqueeze.in_port(0)

            src = port.get_source()
            port.get_connection().set_source(transpose_copy.out_port(0))
            src.connect(start_port)
