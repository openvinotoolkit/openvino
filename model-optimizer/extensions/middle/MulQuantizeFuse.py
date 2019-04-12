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

from mo.graph.graph import Graph, Node
from mo.middle.passes.conv import get_tensor_in_port, get_value_in_port
from mo.middle.replacement import MiddleReplacementPattern


class MulQuantizeFuse(MiddleReplacementPattern):
    """ Fuses Mul --> Quantize sequence if possible
    """
    enabled = False

    def run_after(self):
        return []

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('preop', dict(op='Mul')),
                ('preoped', dict()),
                ('quantize', dict(op='Quantize')),
            ],
            edges=[
                ('preop', 'preoped'),
                ('preoped', 'quantize', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: Dict[str, Node]):
        quantize = match['quantize']
        preop = match['preop']

        # Check for total number of Mul consumers -- if something else consume its output it cannot be fused
        if len(preop.out_node().out_nodes()) > 1:
            log.debug('MulQuantizeFuse: cannot fuse because Mul have multiple consumers')
            return

        # If the fusion is applicable, direct modifications to quantize 1-st and 2-nd inputs
        # are performed. So the data nodes at those inputs shouldn't have more than 1 consumer
        # maximum 2 consumers to the same quantize op (consumed by 1st and 2nd ports).
        # TODO: relax this limitation and duplicate data nodes accordingly to modify the input range freely

        # Provisional limitation that related to binary quantization
        # TODO: Relax it beyond binarization case
        # Provisional limitation that related to binary quantization
        # TODO: Relax it beyond binarization case
        if len(quantize.in_node(1).out_nodes()) != 1 or \
                len(quantize.in_node(2).out_nodes()) != 1 or \
                len(quantize.in_node(3).out_nodes()) != 1 or len(quantize.in_node(4).out_nodes()) != 1 or \
                quantize.levels != 2:
            log.debug('MulQuantizeFuse: cannot fuse because Quantize op has '
                      'unexpected number of consumers for ports 1, 2, 3 or 4')
            return

        tensor_port, value_port = get_tensor_in_port(preop), get_value_in_port(preop)


        # Need to flip output_low and output_high for those elements that have multiplier < 0
        # TODO: need some special processing for values that exactly equal to threshold
        if np.all(value_port.data.get_value() <= 0):
            log.debug('MulQuantizeFuse: cannot fuse because Mul op has non-positive multipliers.')

        quantize.in_port(1).data.set_value(quantize.in_port(1).data.get_value() / value_port.data.get_value())
        quantize.in_port(2).data.set_value(quantize.in_port(2).data.get_value() / value_port.data.get_value())

        # Remove Mul as it no longer needed
        quantize.in_port(0).disconnect()
        tensor_port.get_connection().set_destination(quantize.in_port(0))
