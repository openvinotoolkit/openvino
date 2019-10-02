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

from extensions.middle.CheckForCycle import CheckForCycle
from extensions.middle.DeleteNotExecutable import DeleteNotExecutable
from extensions.ops.elementwise import Mul, Pow
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const


class BinarizeWeightsM1P1(MiddleReplacementPattern):
    """ Convert weights to -1/+1 form

        Applicable for convolutions and other operations that have 'weights' that combined with the input data
        by mean of multiplication operation. So any linear operator suits. Detect such operations by
        multiplication_transparent attribute -- if it is presents and set to True, then multiplication term
        can be passed through the operation. If multiplication_transparent attribute is set to True for an operation,
        such operation should also has multiplication_transparent_ports that contain a list of pairs with
        port indices (in_port, out_port) that defines which port pairs can pass multiplication through.

        For example for some convolutional operation which has 2 ports (input tensor and weights) and 1 output port
        this list includes [(0,0)(1,0)]. If convolutional operation also has biases at port 2, it is not included into
        this list because this port is not transparent for multiplication operation.

        multiplication_transparent_ports can be None if all possible input/output pairs are multiplication
        transparent.

        #TODO Describe how to apply multiplication at output ports -- this is not specified. In the current definition
        we can pass through only scalar multiplication, but we already requre passing it channel-wise.
    """
    enabled = True

    def run_after(self):
        return []

    def run_before(self):
        # CheckForCycle and DeleteNotExecutable run graph clean up which should not be run before weights binarization
        return [CheckForCycle, DeleteNotExecutable]

    def pattern(self):
        return dict(
            nodes=[
                ('quantize', dict(kind='op', op='FakeQuantize')),
                ('quantized', dict()),
                ('operator', dict(kind='op', multiplication_transparent=True)),
            ],
            edges=[
                ('quantize', 'quantized'),
                ('quantized', 'operator'),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        assert match['operator'].has('multiplication_transparent_ports')

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
        quantize = match['quantize']
        output_low = quantize.in_node(3)
        output_high = quantize.in_node(4)

        if not output_low.has_valid('value') and not output_high.has_valid('value'):
            return

        output_low = output_low.value
        output_high = output_high.value

        # This pass is applicable for binarization only. Other intX variants are not relevant.
        if quantize.levels != 2:
            return

        # Recognize two cases: 0/+1 and -1/+1.
        zp1 = np.all(output_low == 0) or np.all(output_high == 0)
        m1p1 = np.all(-output_low == output_high)
        if (not zp1 and not m1p1) or (zp1 and m1p1):
            log.debug('BinarizeWeightsM1P1 cannot apply transformation for data {} because it does\'t has one of'
                      ' 0/+1 or -1/+1 forms.'.format(match['quantized'].name))
            return

        # Recognize scalar
        if len(np.unique(output_low)) != 1 or len(np.unique(output_high)) != 1:
            log.debug('BinarizeWeightsM1P1 cannot apply transformation for data {} because output_low or output_high '
                      'cannot be interpreted as scalars.'.format(match['quantized'].name))
            return

        # TODO: Extract real scalar from 3rd and 4th inputs; reusing original tensors is dangerous because
        #       it may have incompatible shape.

        mult_term = quantize.in_node(3) if np.all(output_high == 0) else quantize.in_node(4)

        # Patch inflow path (by diving by mult_term)
        # Put a new Pow/Mul combination here:
        #       ---->---- (here)---> data ---> [3rd/4th ports]quantize ---> quantized ---> operator

        if len(match['quantized'].out_nodes()) > 1:
            log.debug('BinarizeWeightsM1P1: len(match[\'quantized\'].out_nodes()) > 1')
            return
        power_of_exponent = Const(graph, {'value': np.array(-1.0)}).create_node_with_data()
        div_op = Pow(graph, {'name': quantize.name + '/DivNormalize'})
        div_output = div_op.create_node_with_data([mult_term, power_of_exponent])

        for i in [3, 4]:
            match['quantize'].insert_node_with_data_before(
                match['quantize'].in_node(i),
                Mul,
                dict(name=quantize.name + '/MulNormalize'),
                additional_inputs=[div_output],
            )

        match['quantized'].value = None  # reset value because it will be recomputed
        match['quantize'].infer(match['quantize'])

        # Put a complimentary new Mul node here:   operator -->---(here)-----> operator.out_node()

        match['operator'].insert_node_with_data_after(
            match['operator'].out_node(),
            Mul,
            dict(name=match['operator'].name + '/MulNormalize'),
            [mult_term],
        )

        # Disable 'operator' fusion with linear ops, otherwise it will annihilate changes that we just made
        match['operator']['can_be_fused'] = False
