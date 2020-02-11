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

from extensions.back.ReshapeMutation import ReshapeMutation
from extensions.back.TransposeToPermute import TransposeToPermute
from extensions.ops.MatMul import FullyConnected, Gemm, MatMul
from extensions.ops.transpose import Transpose
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.passes.fusing.helpers import get_tensor_in_port, get_value_in_port
from mo.ops.const import Const
from mo.ops.reshape import Reshape
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

        order = Const(graph, {'value': int64_array(transpose_order)}).create_node()
        transpose = Transpose(graph, {'name': name + '/{}_port_transpose'.format(in_port_idx)}).create_node()

        port_source = in_port.get_source()
        in_port.get_connection().set_source(transpose.out_port(0))
        transpose.in_port(0).connect(port_source)
        transpose.in_port(1).connect(order.out_port(0))

        transpose['override_output_shape'] = True

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['matmul']
        is_v10 = graph.graph['cmd_params'].generate_experimental_IR_V10

        if not is_v10 and node.has_and_set('transpose_a') and node.in_port(0).data.get_value() is not None:
            MatMulConstTransposesExtraction.insert_transpose(node, 0)
            node['transpose_a'] = False

        if not is_v10 and node.has_and_set('transpose_b') and node.in_port(1).data.get_value() is not None:
            MatMulConstTransposesExtraction.insert_transpose(node, 1)
            node['transpose_b'] = False

        if is_v10 and not node.has_and_set('transpose_b'):
            B_shape = node.in_port(1).data.get_shape()
            B_value = node.in_port(1).data.get_value()
            FQ_on_weights = node.in_port(1).get_source().node.has_and_set('stop_value_propagation')
            if (B_value is not None or FQ_on_weights) and B_shape[B_shape != 1].size <= 2:
                MatMulConstTransposesExtraction.insert_transpose(node, 1)
                node['transpose_b'] = True


class MatMulToFullyConnected(BackReplacementPattern):
    """
    All infers are done during replacement because otherwise shape_inference will raise on shape collision,
    but it is appropriate here cause operation semantic change (MatMul->FullyConnected)
    """

    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_before(self):
        return [ReshapeMutation, TransposeToPermute]

    def run_after(self):
        return [MatMulConstTransposesExtraction]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('matmul', dict(kind='op', op='MatMul'))],
            edges=[]
        )

    @staticmethod
    def get_matmul_BIKO(node):
        A_shape, B_shape = MatMul.shape_alignment(node)

        I = A_shape[-2]
        K = A_shape[-1]
        O = B_shape[-1]
        B = A_shape[:-2]

        return B, I, K, O, A_shape, B_shape

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['matmul']
        name = node.soft_get('name', node.id)

        A_shape = node.in_port(0).data.get_shape()
        B_shape = node.in_port(1).data.get_shape()
        out_shape = node.out_port(0).data.get_shape()

        assert A_shape is not None and B_shape is not None and out_shape is not None

        B_value = node.in_port(1).data.get_value()
        if (B_value is not None or node.in_port(1).get_source().node.has_and_set('stop_value_propagation')) and B_shape[
            B_shape != 1].size <= 2:
            # transferring from MatMul representation: [B, I, K] * [B, K, O] = [B, I, O]
            # to FullyConnected representation: [I, K] * [O, K] = [I, O]
            B, I, K, O, aligned_A_shape, aligned_B_shape = MatMulToFullyConnected.get_matmul_BIKO(node)

            # weights normalization
            if not node.transpose_b:
                # FullyConnected weights layout is OI
                # MatMul second input layout is (B)IO
                transpose_order = list(range(B_shape.size))
                transpose_order[-1], transpose_order[-2] = transpose_order[-2], transpose_order[-1]

                order = Const(graph, {'value': int64_array(transpose_order)}).create_node()
                transpose = Transpose(graph, {'name': name + '/weights_transpose'}).create_node()

                weights_source = node.in_port(1).get_source()
                node.in_port(1).get_connection().set_source(transpose.out_port(0))
                transpose.in_port(0).connect(weights_source)
                transpose.in_port(1).connect(order.out_port(0))

                order.infer(order)
                transpose.infer(transpose)

            if node.in_port(1).data.get_shape().size != 2:
                const = Const(graph, {'value': int64_array([-1, K])}).create_node()
                reshape = Reshape(graph, {'name': name + '/weights_reshape'}).create_node()

                weights_source = node.in_port(1).get_source()
                node.in_port(1).get_connection().set_source(reshape.out_port(0))

                reshape.in_port(0).connect(weights_source)
                reshape.in_port(1).connect(const.out_port(0))

                const.infer(const)
                reshape.infer(reshape)

            assert np.all(np.array_equal(node.in_port(1).data.get_shape(), int64_array([O, K]))), \
                "MatMul `{}` was not converted to FullyConnected: wrong weights shape: {}, " \
                "B={}, I={}, K={}, O={}".format(name, node.in_port(1).data.get_shape(), B, I, K, O)

            node.in_port(1).bin = 'weights'
            del node['transpose_b']

            # input normalization
            if node.transpose_a:
                transpose_order = list(range(A_shape.size))
                transpose_order[-1], transpose_order[-2] = transpose_order[-2], transpose_order[-1]

                order = Const(graph, {'value': int64_array(transpose_order)}).create_node()
                transpose = Transpose(graph, {'name': name + '/input_transpose'}).create_node()

                input_source = node.in_port(0).get_source()
                node.in_port(0).get_connection().set_source(transpose.out_port(0))
                transpose.in_port(0).connect(input_source)
                transpose.in_port(1).connect(order.out_port(0))

                order.infer(order)
                transpose.infer(transpose)

            if A_shape.size != 2:
                const = Const(graph, {'value': int64_array([-1, K])}).create_node()
                reshape = Reshape(graph, {'name': name + '/input_reshape'}).create_node()

                input_source = node.in_port(0).get_source()
                node.in_port(0).get_connection().set_source(reshape.out_port(0))
                reshape.in_port(0).connect(input_source)
                reshape.in_port(1).connect(const.out_port(0))

                const.infer(const)
                reshape.infer(reshape)

            assert np.all(np.array_equal(node.in_port(0).data.get_shape(), int64_array([np.prod(B) * I, K]))), \
                "MatMul `{}` wasn't converted to FullyConnected: wrong input shape: {}, " \
                "B={}, I={}, K={}, O={}".format(name, node.in_port(0).data.get_shape(), B, I, K, O)

            del node['transpose_a']

            FullyConnected.update_node_stat(node, {'out-size': O})

            # output normalization
            if out_shape.size != 2:
                const = Const(graph, {'value': int64_array([*B, I, O])}).create_node()
                reshape = Reshape(graph, {'name': name + '/output_reshape'}).create_node()

                dst = node.out_port(0).get_destination()
                node.out_port(0).get_connection().set_destination(reshape.in_port(0))
                const.out_port(0).connect(reshape.in_port(1))
                reshape.out_port(0).connect(dst)

                node.infer(node)

                const.infer(const)
                reshape.infer(reshape)

        else:
            assert A_shape.size == out_shape.size
            assert B_shape.size <= out_shape.size
            if B_shape.size != out_shape.size:
                unsqueeze_dim = Const(graph, {'value': int64_array(list(range(out_shape.size - B_shape.size)))
                                              }).create_node()
                unsqueeze = Unsqueeze(graph, {}).create_node()
                B_source = node.in_port(1).get_source()
                node.in_port(1).get_connection().set_source(unsqueeze.out_port(0))
                unsqueeze.in_port(0).connect(B_source)
                unsqueeze.in_port(1).connect(unsqueeze_dim.out_port(0))

                unsqueeze_dim.infer(unsqueeze_dim)
                unsqueeze.infer(unsqueeze)

            Gemm.update_node_stat(node, {
                'transpose_a': node.has_and_set('transpose_a'),
                'transpose_b': node.has_and_set('transpose_b'),
            })


class SSBiasAddonForFC(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [MatMulToFullyConnected]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('fc', dict(kind='op', op='FullyConnected')),
                ('fc_d', dict(kind='data')),
                ('scale_shift', dict(kind='op', op='ScaleShift')),
            ],
            edges=[
                ('fc', 'fc_d'),
                ('fc_d', 'scale_shift'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['fc']
        ss = match['scale_shift']
        name = node.soft_get('name', node.id)

        weights_value = ss.in_port(1).data.get_value()
        assert weights_value is not None
        if not np.all(weights_value == 1):
            return
        out_size = node.soft_get('out-size', None)
        assert out_size is not None, \
            "FullyConnected should have `out-size` parameter, but it doesn't for node {}".format(name)
        shift_shape = ss.in_port(2).data.get_shape()

        if not np.array_equal(int64_array([out_size]), shift_shape):
            return

        node.add_input_port(2, skip_if_exist=True)
        ss.in_port(2).get_connection().set_destination(node.in_port(2))
        ss.out_port(0).get_connection().set_source(ss.in_port(0).get_source())


class BiasAddonForFC(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]
    force_clean_up = True

    def run_after(self):
        return [MatMulToFullyConnected]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('fc', dict(kind='op', op='FullyConnected')),
                ('fc_d', dict(kind='data')),
                ('add', dict(kind='op', op='Add')),
            ],
            edges=[
                ('fc', 'fc_d'),
                ('fc_d', 'add'),
            ]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['fc']
        name = node.soft_get('name', node.id)
        add = match['add']

        if 2 in node.in_ports() and not node.in_port(2).disconnected():
            return

        out_size = node.soft_get('out-size', None)
        assert out_size is not None, \
            "FullyConnected should have `out-size` parameter, but it doesn't for node {}".format(name)

        tensor_port, value_port = get_tensor_in_port(add), get_value_in_port(add)
        if value_port is None:
            return

        shift_shape = value_port.data.get_shape()
        if not any([np.array_equal(int64_array(suitable_shape), shift_shape)
                    for suitable_shape in [[1, out_size], [1, 1], [out_size], [1], []]]):
            return

        broadcasted_value = np.broadcast_to(value_port.data.get_value(), [1, out_size])
        const = Const(graph, {'name': name + '/Bias_', 'value': broadcasted_value}).create_node()

        node.add_input_port(2, skip_if_exist=True)
        const.out_port(0).connect(node.in_port(2))
        add.out_port(0).get_connection().set_source(tensor_port.get_source())
        node.in_port(2).bin = 'biases'


class FullyConnectedFinalization(BackReplacementPattern):
    enabled = True
    graph_condition = [
        lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10,
    ]
    force_clean_up = True

    def run_after(self):
        return [SSBiasAddonForFC, BiasAddonForFC, PullTransposeThroughFQUp]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('fc', dict(kind='op', op='FullyConnected'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['fc']
        name = node.soft_get('name', node.id)
        out_size = node.soft_get('out-size', None)

        assert out_size is not None, \
            "FullyConnected should have `out-size` parameter, but it doesn't for node {}".format(name)

        in_ports = node.in_ports()

        # [I, K] * [O, K] = [I, O]
        if 1 not in in_ports or node.in_port(1).disconnected():
            # add fake weights
            input_shape = node.in_port(0).data.get_shape()
            assert input_shape is not None
            K = input_shape[-1]
            node.add_input_port(1, skip_if_exist=True)
            const = Const(graph, {'value': np.ones([out_size, K])}).create_node()
            node.in_port(1).connect(const.out_port(0))
            node.in_port(1).bin = 'weights'

        if 2 not in in_ports or node.in_port(2).disconnected():
            # add fake biases
            node.add_input_port(2, skip_if_exist=True)
            const = Const(graph, {'value': np.zeros([out_size])}).create_node()
            node.in_port(2).connect(const.out_port(0))
            node.in_port(2).bin = 'biases'

        bias_reshape = create_op_node_with_second_input(
            graph, Reshape, int64_array([-1]), {'name': name + '/1D_bias_', 'override_output_shape': True},
            node.in_port(2).get_source().node
        )
        node.in_port(2).get_connection().set_source(bias_reshape.out_port(0))


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
        return [MatMulToFullyConnected]

    def run_before(self):
        return [TransposeToPermute]

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
