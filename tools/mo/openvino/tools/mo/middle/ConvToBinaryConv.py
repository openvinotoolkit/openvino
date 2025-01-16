# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from openvino.tools.mo.ops.elementwise import Mul, Add
from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.tf.graph_utils import create_op_node_with_second_input
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class ConvToBinaryConv(MiddleReplacementPattern):
    """ Transform usual convolution with [0,+1] input and [-1,+1] to BinaryConvolution

        Modifies output terms after the Convolution to be able to apply BinaryConvolution
        operation instead that accepts [-1,1] input and [-1,1] weights. It requires modification
        channel-wise addition with weights reduced along all axis except output channel dimension.
    """
    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                # This pass is applicable for binarization only. Other intX variants are not relevant.
                ('quantize', dict(kind='op', op='FakeQuantize', levels=2)),
                ('quantized', dict()),  # input tensor, not weights
                ('operator', dict(kind='op', type='Convolution')),
            ],
            edges=[
                ('quantize', 'quantized'),
                ('quantized', 'operator', {'in':0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        assert match['operator'].has('multiplication_transparent_ports')

        quantize = match['quantize']

        port = match['operator'].input_ports_with(match['quantized'])
        assert len(port) >= 1
        if len(port) > 1:
            log.debug('BinarizeWeightsM1P1 cannot apply transformation for data {} because it consumed more'
                      ' than once'.format(match['quantized'].name))
            return

        assert len(port) == 1
        port = port[0]
        applicable = [pair for pair in match['operator'].multiplication_transparent_ports if pair[0] == port]
        if len(applicable) == 0:
            return

        # Look at 3-rd and 4-th inputs of FakeQuantize -- they have constants that should be passed through.
        # Assume that the constant that should be passed through is a scalar.
        output_low = quantize.in_node(3)
        output_high = quantize.in_node(4)
        assert len(output_low.out_nodes()) == 1
        assert len(output_high.out_nodes()) == 1

        if not output_low.has_valid('value') and not output_high.has_valid('value'):
            return

        output_low = output_low.value
        output_high = output_high.value

        operator = match['operator']

        weights = operator.in_node(1).value
        weights_rounded = np.round(weights)
        weights_consistent = np.all(np.isclose(weights, weights_rounded)) and \
                             set(np.unique(weights_rounded)).issubset({-1, 1})

        if weights_consistent and np.all(np.isclose(output_low, 0)) and np.all(np.isclose(output_high, 1)):
            reduction_indices = set(range(len(weights.shape))) - set([operator.output_feature_channel])
            weights_reduced = np.add.reduce(weights, axis=tuple(reduction_indices))
            weights_reduced = weights_reduced.reshape([len(weights_reduced), 1, 1])  # FIXME: works for NCHW only

            operator_name = operator.soft_get('name', operator.id)
            add = create_op_node_with_second_input(graph, Add, weights_reduced, {'name': operator_name + '/Add_'})
            mul = create_op_node_with_second_input(graph, Mul, mo_array(0.5), {'name': operator_name + '/Mul_'})

            add.out_port(0).connect(mul.in_port(0))

            operator.out_port(0).get_connection().set_source(mul.out_port(0))
            add.in_port(0).connect(operator.out_port(0))

            operator['pad_value'] = float(-1.0)
        elif weights_consistent and np.all(np.isclose(output_low, -1)) and np.all(np.isclose(output_high, +1)):
            pass
        else:
            log.debug('ConvToBinaryConv: cannot apply transformation because input range is neither in [0, +1] nor '
                      'in [-1, +1].')
            return

        operator['type'] = 'BinaryConvolution'
        operator['mode'] = 'xnor-popcount'
        operator['pad_value'] = operator.soft_get('pad_value', float(0))
        operator['input'] = operator.in_node(0).shape[1]
        # Weights are not bit-packed yet; there should be a separate transformation to do that

        assert output_low.size == 1
        assert output_high.size == 1

        output_low = quantize.in_node(3)
        output_high = quantize.in_node(4)

        # Make sure that low/high values are exactly 0/1
        output_low.value = np.zeros(output_low.shape, dtype=np.float32)
        output_high.value = np.ones(output_high.shape, dtype=np.float32)
