# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.ops.op import Op


class FFTBase(Op):
    enabled = False
    op = None
    version = 'opset7'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'out_ports_count': 1,
            'in_ports_count': 3,
            'version': self.version,
            'infer': self.infer
        }
        super().__init__(graph, mandatory_props, attrs)

    def infer(self, node: Node):
        node_name = node.soft_get(node.name, node.id)
        assert len([p for p in node.in_ports().values() if not p.disconnected()]) in [2, 3], \
            '(I)DFT node {} must have 2 or 3 inputs'.format(node_name)

        src_shape = node.in_port(0).data.get_shape()
        assert src_shape is not None, 'The input data shape of (I)DFT node {} must not be None'.format(node_name)
        assert src_shape[-1] == 2, \
            'The last dimension of input shape of (I)DFT node {} should be equal to 2'.format(node_name)

        input_rank = len(src_shape)
        assert input_rank >= 2, 'The input rank of (I)DFT node {} should be greater or equal to 2'.format(node_name)

        axes = FFTBase.get_axes(node)
        assert input_rank >= len(axes) + 1, \
            'The input rank must be greater than number of (I)DFT node {} axes'.format(node_name)
        axes = FFTBase.canonicalize_axes(axes, input_rank)
        assert (input_rank - 1) not in axes, '(I)DFT node {} axes cannot contain the last axis'.format(node_name)
        assert len(set(axes)) == len(axes), '(I)DFT node {} axes must be unique.'.format(node_name)

        output_shape = int64_array(src_shape)
        if node.is_in_port_connected(2):
            signal_size = FFTBase.get_signal_size(node)
            signal_size = FFTBase.canonicalize_signal_size(signal_size, axes, src_shape)
            output_shape[axes] = signal_size

        node.out_port(0).data.set_shape(output_shape)

    @staticmethod
    def canonicalize_axes(axes, input_rank):
        """
        FFT operation supports for negative axes to transform. More precisely, according to the FFT operation
        specification, axes should be integers from -(r - 1) to (r - 2) inclusively, where r = rank(data).
        A negative axis 'a' is interpreted as an axis 'r - 1 + a'. The reason is the following: real input
        tensor of the shape [n_0, ..., n_{r - 1}, 2] is interpreted as a complex tensor with the shape
        [n_0, ..., n_{r - 1}]. Hence, we need to 'canonicalize' axes using the formula 'r - 1 + a'.

        :param axes: axes to canonicalize
        :param input_rank: input tensor rank
        :return: canonicalized axes
        """
        result = axes.copy()
        for i, axis in enumerate(axes):
            if axis < 0:
                result[i] = axis + input_rank - 1
        return result

    @staticmethod
    def canonicalize_signal_size(signal_size, axes, input_shape):
        result = signal_size.copy()
        for i, axis in enumerate(axes):
            size = signal_size[i]
            if size == -1:
                result[i] = input_shape[axis]
        return result

    @staticmethod
    def get_axes(node: Node):
        axes = node.in_port(1).get_source().data.get_value()
        node_name = node.soft_get('name', node.id)
        assert axes is not None, 'The input with axes is not constant for node {}'.format(node_name)
        return int64_array(axes)

    @staticmethod
    def get_signal_size(node: Node):
        src_shape = node.in_port(0).data.get_shape()
        assert src_shape is not None
        input_rank = len(src_shape)
        if node.is_in_port_connected(2):
            signal_size = node.in_port(2).get_source().data.get_value()
        else:
            axes = FFTBase.get_axes(node)
            signal_size = [src_shape[: input_rank - 1][a] for a in axes]

        node_name = node.soft_get('name', node.id)
        assert signal_size is not None, 'The input with signal_size is not constant for node {}'.format(node_name)

        return int64_array(signal_size)


class DFT(FFTBase):
    op = 'DFT'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
        }
        mandatory_props.update(attrs)
        super().__init__(graph, mandatory_props)


class IDFT(FFTBase):
    op = 'IDFT'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
        }
        mandatory_props.update(attrs)
        super().__init__(graph, mandatory_props)
