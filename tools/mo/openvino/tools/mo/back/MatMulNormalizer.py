# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.back.TransposeReduceFusing import TransposeReduce
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.back.replacement import BackReplacementPattern
from openvino.tools.mo.front.caffe.extractors.utils import get_canonical_axis_index
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph, Node
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.shape import Shape
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.shape import node_to_get_shape_value_of_indices, new_shape_node_from_shape_nodes


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

        transpose = create_op_node_with_second_input(graph, Transpose, int64_array(transpose_order),
                                                     {'name': name + '/{}_port_transpose'.format(in_port_idx)})

        port_source = in_port.get_source()
        in_port.get_connection().set_source(transpose.out_port(0))
        transpose.in_port(0).connect(port_source)

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
    r"""
        BEFORE                                      AFTER
    Const                                             Const
        \ \ | / /                                       |
        FakeQuantize                                    T  T T  T  T
            |                                            \ \ | / /
        Transpose                                       FakeQuantize
            |                                                |
         next_op                                          next_op
        `T` is Transpose for short
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        # in case FQ->Transpose->Reduce we should first try to optimize out Transpose
        return [MatMulConstTransposesExtraction, TransposeReduce]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('fq_const_input', dict(kind='op', type='Const')),
                ('fq_const_input_d', dict()),
                ('fq', dict(kind='op', type='FakeQuantize')),
                ('fq_d', dict()),
                ('transpose', dict(kind='op', type='Transpose')),
            ],
            edges=[
                ('fq_const_input', 'fq_const_input_d'),
                ('fq_const_input_d', 'fq', {'in': 0}),
                ('fq', 'fq_d'),
                ('fq_d', 'transpose'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        fq = match['fq']

        if len(fq.out_port(0).get_destinations()) > 1:
            # FQ should have only one child -- Transpose for optimization
            return

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


class SmartReshape_HC_Reshape_MatMul(BackReplacementPattern):
    r"""
    Relaxes hard-coded input of Reshape in such sub-graphs:

    input_1     Constant
        \       /
        Reshape    input_2
           \       /
              MatMul
                |
    """
    enabled = True
    force_clean_up = True

    def run_after(self):
        return [MatMulConstTransposesExtraction]

    def pattern(self):
        return dict(
            nodes=[
                ('output_shape', dict(type='Const')),
                ('output_shape_d', dict()),
                ('reshape', dict(type='Reshape')),
                ('reshape_d', dict()),
                ('other_input', dict(type=lambda t: t not in ['Reshape', 'Transpose'])),
                ('other_input_d', dict()),
                ('matmul', dict(type='MatMul')),
            ],
            edges=[
                ('output_shape', 'output_shape_d'),
                ('output_shape_d', 'reshape', {'in': 1}),
                ('reshape', 'reshape_d'),
                ('reshape_d', 'matmul'),
                ('other_input', 'other_input_d'),
                ('other_input_d', 'matmul'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        matmul = match['matmul']
        reshape = match['reshape']
        other_input_port_idx = 0 if match['matmul'].in_port(0).get_source().node.id == match['other_input'].id else 1
        shape_source = match['matmul'].in_port(other_input_port_idx).get_source()
        initial_reshape_pattern = reshape.in_port(1).data.get_value()
        if len(initial_reshape_pattern) != 2:
            return

        reshape_is_A_input = matmul.in_port(0).get_source().node.id == reshape.id
        if reshape_is_A_input:
            idx = -1 if matmul.transpose_b else -2
        else:
            idx = -2 if matmul.transpose_a else -1
        idx = get_canonical_axis_index(initial_reshape_pattern, idx)

        shape_name = shape_source.node.soft_get('name', shape_source.node.id)
        shape = Shape(graph, {'name': shape_name + '/Shape'}).create_node()
        shape.in_port(0).connect(shape_source)
        C = node_to_get_shape_value_of_indices(shape, [idx])
        N = Const(graph, {'name': shape_name + '/MinusOne', 'value': int64_array([-1])}).create_node()

        if len(initial_reshape_pattern) == 2:
            if reshape_is_A_input:
                reshape_pattern = [C, N] if matmul.transpose_a else [N, C]
            else:
                reshape_pattern = [N, C] if matmul.transpose_b else [C, N]
            new_reshape_pattern = new_shape_node_from_shape_nodes(reshape_pattern)
            reshape.in_port(1).get_connection().set_source(new_reshape_pattern.out_port(0))
        else:
            return

