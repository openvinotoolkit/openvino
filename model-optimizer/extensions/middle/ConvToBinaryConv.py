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

import logging as log

import numpy as np

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from extensions.ops.elementwise import Mul, Add


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

        if np.all(np.isclose(output_low, 0)) and np.all(np.isclose(output_high, 1)):

            weights = operator.in_node(1).value
            reduction_indices = set(range(len(weights.shape))) - set([operator.output_feature_channel])
            weights_reduced = np.add.reduce(weights, axis=tuple(reduction_indices))
            weights_reduced = weights_reduced.reshape([len(weights_reduced), 1, 1])

            add_term = Const(graph, {'value': weights_reduced}).create_node()
            add = Add(graph, {}).create_node()
            add.in_port(1).connect(add_term.out_port(0))
            mul_term = Const(graph, {'value': np.array(0.5)}).create_node()
            mul = Mul(graph, {}).create_node()
            mul.in_port(1).connect(mul_term.out_port(0))
            add.out_port(0).connect(mul.in_port(0))

            operator.out_port(0).get_connection().set_source(mul.out_port(0))
            add.in_port(0).connect(operator.out_port(0))

            operator['pad_value'] = float(-1.0)
        elif np.all(np.isclose(output_low, -1)) and np.all(np.isclose(output_high, +1)):
            pass
        else:
            log.debug('ConvToBinaryConv: cannot apply transformation because input range is neither in [0, +1] nor '
                      'in [-1, +1].')
            return

        operator['type'] = 'BinaryConvolution'
        operator['mode'] = 'xnor-popcount'
        operator['input'] = operator.in_node(0).shape[1]
        # Weights are not bit-packed yet; there should be a separate transformation to do that

        assert output_low.size == 1
        assert output_high.size == 1

        output_low = quantize.in_node(3)
        output_high = quantize.in_node(4)

        # Make sure that low/high values are exactly 0/1
        output_low.value = np.zeros(output_low.shape)
        output_high.value = np.ones(output_high.shape)
