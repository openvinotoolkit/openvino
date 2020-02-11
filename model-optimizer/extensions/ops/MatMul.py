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

import logging as log

import numpy as np

from mo.front.common.partial_infer.utils import assign_dims_to_weights, int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class MatMul(Op):
    """
    MatMul operation takes two tensors and performs usual matrix-matrix multiplication, matrix-vector multiplication
    or vector-matrix multiplication depending on argument shapes.

    Input tensors can have any rank >= 1.

    Two right-most axes in each tensor are interpreted as matrix rows and columns dimensions while
    all left-most axes (if present) are interpreted as multi-dimensional batch:

    [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, ROW_INDEX_DIM, COL_INDEX_DIM]

    The operation supports usual broadcast semantics for batch dimensions.
    It enables multiplication of batch of pairs of matrices in a single shot.

    Before matrix multiplication, there is an implicit shape alignment for input arguments.
    It consists of the following steps:

    1. If rank of an input less than 2 it is unsqueezed to 2D tensor by adding axes with size 1 to the left of the shape
        For example, if input has shape [S] it will be reshaped to [1, S]. It is applied for each input independently
    2. Applied transpositions specified by optional transpose_a and transpose_b attributes
    3. If ranks of input arguments are different after steps 1 and 2, each is unsqueezed from the left side of
        the shape by necessary number of axes to make both shapes of the same rank
    4. Usual rules of the broadcasting are applied for batch dimensions

    Two attributes, transpose_a and transpose_b specifies embedded transposition for two right-most dimension for the
    first and the second input tensors correspondingly. It implies swapping of ROW_INDEX_DIM and COL_INDEX_DIM in
    the corresponding input tensor. Batch dimensions are not affected by these attributes.

    Shape inference mechanism:
        0-port aligned input shape:
            [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, A_ROW_INDEX_DIM, A_COL_INDEX_DIM]
        1-port aligned input shape:
            [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, B_ROW_INDEX_DIM, B_COL_INDEX_DIM]
        where A_COL_INDEX_DIM == B_ROW_INDEX_DIM

        Output shape:
            [BATCH_DIM_1, BATCH_DIM_2,..., BATCH_DIM_K, A_ROW_INDEX_DIM, B_COL_INDEX_DIM]

    """
    op = 'MatMul'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'transpose_a': False,
            'transpose_b': False,
            'infer': __class__.infer,
            'in_ports_count': 2,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def supported_attrs(self):
        return [
            'transpose_a',
            'transpose_b',
        ]

    @staticmethod
    def shape_alignment(node: Node):
        """
        Specification of MatMul operation allows inputs to be aligned together before matrix multiplication.
        Alignment steps described in MatMul operation doc-string upper in current file.

        Current method raises an error if input shapes are not valid at any step of alignment process
        :return: aligned copies of both input shapes
        """
        node_name = node.soft_get('name', str(node.id))
        input_shapes = [node.in_port(i).data.get_shape() for i in range(2)]
        transpose_a = node.has_and_set('transpose_a')
        transpose_b = node.has_and_set('transpose_b')

        transformed_shapes = []
        for i, shape in enumerate(input_shapes):
            input_shape = shape.copy()
            # prerequisites check
            assert input_shape is not None, "MatMul has shape=`None` for {} input of `{}` node".format(i, node_name)
            assert input_shape.ndim == 1, "MatMul doesn't support scalar inputs. {} input of `{}` node has shape {}" \
                                          "".format(i, node_name, input_shape)
            assert input_shape.size >= 1, "MatMul doesn't support inputs with rank lower than 1. {} input of `{}` " \
                                          "node has shape {}".format(i, node_name, input_shape)

            # shape alignment
            if input_shape.size < 2:
                input_shape = np.insert(input_shape, 0, 1)
            if (i == 0 and transpose_a) or (i == 1 and transpose_b):
                input_shape[-2], input_shape[-1] = input_shape[-1], input_shape[-2]

            max_shape_length = max(input_shapes[0].size, input_shapes[1].size)
            input_shape = np.insert(input_shape, 0, [1] * (max_shape_length - input_shape.size))
            transformed_shapes.append(input_shape)

        A_shape = transformed_shapes[0]
        B_shape = transformed_shapes[1]

        assert A_shape.size == B_shape.size, \
            "Shapes were not aligned by length for MatMul `{}`. Shapes: `{}`".format(node_name, transformed_shapes)

        # batch broadcasting
        batch_len = A_shape.size - 2
        for i in range(batch_len):
            if A_shape[i] != B_shape[i]:
                if A_shape[i] == 1:
                    A_shape[i] = B_shape[i]
                if B_shape[i] == 1:
                    B_shape[i] = A_shape[i]

        assert np.array_equal(A_shape[:-2], B_shape[:-2]), \
            "MatMul input shapes are incorrect. BATCH_DIMs are not equal. Node: {}. Aligned shapes: {}" \
            "".format(node_name, transformed_shapes)

        return A_shape, B_shape

    @staticmethod
    def infer(node: Node):
        """
        Performs shape inference of MatMul node as operation doc-string says
        Raises on any shape inconsistency
        """
        name = node.soft_get('name', str(node.id))
        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) == 2 and 0 in connected_in_ports and 1 in connected_in_ports, \
            "MatMul should have 2 connected input ports, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_in_ports)

        log.debug('MatMul `{}` input shapes: {}'.format(name, [node.in_port(i).data.get_shape() for i in range(2)]))
        A_shape, B_shape = MatMul.shape_alignment(node)
        log.debug('MatMul `{}` aligned input shapes: {}'.format(name, [A_shape, B_shape]))

        assert A_shape[-1] == B_shape[-2], \
            "MatMul input shapes are incorrect. COL_INDEX_DIMs are not equal. Node: {}. Shapes: {}" \
            "".format(name, [A_shape, B_shape])

        output_shape = np.concatenate((A_shape[:-1], B_shape[-1:]))
        node.out_port(0).data.set_shape(output_shape)

        in_ch = 0 if not node.transpose_b else 1
        out_ch = 1 if not node.transpose_b else 0
        assign_dims_to_weights(node.in_node(1), None, in_ch, out_ch, node.in_port(1).data.get_shape().size)


# MatMul-like operation from frameworks

class GemmONNX(Op):
    """
    Represents Gemm operation from ONNX

    Missing `type` and `infer` attributes on purpose - node should be decomposed on front phase
    and should never be inferred or translated to IR as is
    """
    op = 'Gemm'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'op': __class__.op,
            'transpose_a': False,
            'transpose_b': False,
            'alpha': 1,
            'beta': 1,
            'broadcast_c': True,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)


class FullyConnected(Op):
    # TODO: remove `infer`, `type` and supported_attrs after op removal from IR Spec
    op = 'FullyConnected'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': __class__.op,
            'type': __class__.op,
            'infer': __class__.infer,
            'in_ports_count': 3,
            'out_ports_count': 1,
        }, attrs)

    def supported_attrs(self):
        return [
            'out-size',
        ]

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) >= 2 and 0 in connected_in_ports and 1 in connected_in_ports, \
            'FullyConnected should have 2 connected input ports, but it doesn\'t for node: `{}`. Ports: {}' \
            ''.format(name, connected_in_ports)

        assert node.has_valid('out-size')
        input_shape = node.in_port(0).data.get_shape()
        weights_shape = node.in_port(1).data.get_shape()
        assert input_shape is not None and weights_shape is not None, \
            'Incorrect FullyConnected input shapes. Node: {}. Shapes: {}'.format(name, [input_shape, weights_shape])
        assert weights_shape.size == 2
        out_size = node.soft_get('out-size')
        assert weights_shape[0] == out_size, 'weights_shape={}, out-size={}'.format(weights_shape, out_size)

        if 2 in connected_in_ports:
            bias_value = node.in_port(2).data.get_value()
            bias_shape = node.in_port(2).data.get_shape()
            assert bias_shape is not None, 'Shape was not inferred for biases of FullyConnected {}'.format(name)
            assert bias_value is not None, 'Value was not inferred for biases of FullyConnected {}'.format(name)
            assert np.array_equal(bias_shape, [out_size]) or np.array_equal(bias_shape, [1, out_size]), \
                'Incorrect FullyConnected bias shape `{}` for node {}. `out-size`={}'.format(bias_shape, node, out_size)

        out_shape = int64_array([*input_shape[:-1], out_size])
        node.out_port(0).data.set_shape(out_shape)


# MatMul-like operations for IR V6

class Gemm(MatMul):
    """
    Represents GEMM operation that is acceptable to appear in v6 IRs
    Inherits MatMul semantic to be re-inferred in back phase and to be successfully translated to IR (v6)
    """
    op = 'GEMM'
    enabled = False
