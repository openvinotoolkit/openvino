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
from typing import Dict
import numpy as np

from extensions.middle.BinarizeWeightsM1P1 import BinarizeWeightsM1P1
from extensions.middle.MulFakeQuantizeFuse import resolve_shared_inputs
from mo.graph.graph import Graph, Node
from mo.middle.replacement import MiddleReplacementPattern


class ReluFakeQuantizeMark(MiddleReplacementPattern):
    """
    This pass marks Relu operations that can be fused to FakeQuantize op with `removable_before_quantize` flag.

    1. We count the number of Relu outputs that are Quantize and can absorb Relu (`quantized_to_fuse_count` attribute).
    2. Relu is fusible if all its outputs can absorb it.

    """
    enabled = True

    def run_after(self):
        return [BinarizeWeightsM1P1]

    def run_before(self):
        from extensions.middle.SharedWeightsDuplication import SharedWeightsDuplication
        return [SharedWeightsDuplication]

    def pattern(self):
        return dict(
            nodes=[
                ('relu', dict(op='ReLU')),
                ('relu_d', dict()),
                ('quantize', dict(op='FakeQuantize', keep_in_IR=True)),
            ],
            edges=[
                ('relu', 'relu_d'),
                ('relu_d', 'quantize', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        relu = match['relu']
        quantize = match['quantize']

        if not relu.has_valid('quantized_to_fuse_count'):
            relu['quantized_to_fuse_count'] = 0

        if quantize.in_node(1).id == quantize.in_node(2).id:
            # Provisional limitation that related to binary quantization
            assert quantize.has_valid('levels') and quantize.levels == 2

            threshold = quantize.in_port(1).data.get_value()
            if threshold is None:
                log.debug('ReluQuantizeFuse: cannot fuse because FakeQuantize op has dynamic input on the 1st port. '
                          'levels=`{}`'.format(quantize.levels))
                return

            relu['quantized_to_fuse_count'] += 1

        else:
            assert quantize.has_valid('levels') and quantize.levels != 2
            min_value = quantize.in_port(1).data.get_value()
            if min_value is None:
                log.debug('ReluQuantizeFuse: cannot fuse because FakeQuantize op has dynamic input on the 1st port, '
                          'levels=`{}`'.format(quantize.levels))
                return
            if np.all(min_value >= 0):
                relu['quantized_to_fuse_count'] += 1

        relu['removable_before_quantize'] = relu['quantized_to_fuse_count'] == len(relu.out_port(0).get_destinations())


class ClampQuantizeMark(MiddleReplacementPattern):
    """
    This pass marks Clamp operations that can be fused to FakeQuantize op with `removable_before_quantize` flag.

    1. We count the number of Clamp outputs that are FakeQuantize and can absorb Clamp (`quantized_to_fuse_count` attribute)
    2. Clamp is fusible if all its outputs can absorb it.

    """
    enabled = True

    def run_after(self):
        return [BinarizeWeightsM1P1]

    def run_before(self):
        from extensions.middle.SharedWeightsDuplication import SharedWeightsDuplication
        return [SharedWeightsDuplication]

    def pattern(self):
        return dict(
            nodes=[
                ('clamp', dict(op='Clamp')),
                ('clamp_d', dict()),
                ('quantize', dict(op='FakeQuantize', keep_in_IR=True)),
            ],
            edges=[
                ('clamp', 'clamp_d'),
                ('clamp_d', 'quantize', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        clamp = match['clamp']
        quantize = match['quantize']
        clamp_min, clamp_max = clamp['min'], clamp['max']

        if not clamp.has_valid('quantized_to_fuse_count'):
            clamp['quantized_to_fuse_count'] = 0

        if quantize.in_node(1).id == quantize.in_node(2).id:
            # Binary case is not tested so we won't fuse Clamp
            assert quantize.has_valid('levels') and quantize.levels == 2
            clamp['removable_before_quantize'] = False
        else:
            assert quantize.has_valid('levels') and quantize.levels != 2
            min_value = quantize.in_port(1).data.get_value()
            if min_value is None:
                log.debug('ReluQuantizeFuse: cannot fuse because FakeQuantize op has dynamic input on the 1st port, '
                          'levels=`{}`'.format(quantize.levels))
                return
            max_value = quantize.in_port(2).data.get_value()
            if max_value is None:
                log.debug('ReluQuantizeFuse: cannot fuse because FakeQuantize op has dynamic input on the 2st port, '
                          'levels=`{}`'.format(quantize.levels))
                return
            if np.all(min_value >= clamp_min) and np.all(max_value <= clamp_max):
                clamp['quantized_to_fuse_count'] += 1

        clamp['removable_before_quantize'] = clamp['quantized_to_fuse_count'] == len(clamp.out_port(0).get_destinations())


class ReluQuantizeFuse(MiddleReplacementPattern):
    """ Fuses ReLU --> FakeQuantize sequence if possible

        Relu --> FakeQuantize fusion is possible if:
            1. Relu is consumed to 0-th port of FakeQuantize
            2. FakeQuantize ports 1 and 2 defines such input range that 0 is not included
    """
    enabled = True

    def run_after(self):
        return [ReluFakeQuantizeMark]

    def run_before(self):
        from extensions.middle.SharedWeightsDuplication import SharedWeightsDuplication
        return [SharedWeightsDuplication]

    def pattern(self):
        return dict(
            nodes=[
                ('relu', dict(removable_before_quantize=True)),
                ('relu_d', dict()),
                ('quantize', dict(op='FakeQuantize', keep_in_IR=True)),
            ],
            edges=[
                ('relu', 'relu_d'),
                ('relu_d', 'quantize', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        quantize = match['quantize']

        if quantize.levels == 2:
            # extra logic due to special 1 & 2 port input meaning in binary case - it is threshold separating two quants
            threshold = quantize.in_port(1).data.get_value()

            # Direct modifications to quantize 1-st port input are performed.
            # So the data node at this input shouldn't have more than 1 consumer maximum 2 consumers to the same
            # quantize op (consumed by 1st and 2nd ports). So we duplicate FakeQuantize in_port 1 data if needed
            resolve_shared_inputs(node=quantize, port_ids_to_duplicate=[1])

            # As we restricted to binarization case only, so we need to detect from
            # which side of 0 FakeQuantize threshold resides:
            #   if the threshold > 0, it remains the same;
            #   if the threshold == 0, it also remains the same;
            #   if the threshold < 0, it should be modified to -infinity that means that all inputs map to output_high
            modification_mask = threshold < 0
            threshold[modification_mask] = float('-inf')

        # Reconnect ReLU as it no longer needed for current FakeQuantize
        in_relu_connection = quantize.in_port(0).get_source().node.in_port(0).get_connection()
        quantize.in_port(0).disconnect()
        in_relu_connection.add_destination(quantize.in_port(0))
