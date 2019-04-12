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

from extensions.middle.BinarizeWeightsM1P1 import BinarizeWeightsM1P1
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class ReluQuantizeFuse(MiddleReplacementPattern):
    """ Fuses ReLU --> Quantize sequence if possible

        Relu --> Quantize fusion is possible if:
            1. Relu is consumed to 0-th port of Quantize
            2. Quantize ports 1 and 2 defines such input range that 0 is not included
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
                ('relu', dict(op='Relu')),
                ('relued', dict()),
                ('quantize', dict(op='Quantize')),
            ],
            edges=[
                ('relu', 'relued'),
                ('relued', 'quantize', {'in': 0}),
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):

        quantize = match['quantize']

        # Check for total number of ReLU consumers -- if something else consume its output it cannot be fused
        if len(match['relu'].out_node().out_nodes()) > 1:
            log.debug('ReluQuantizeFuse: cannot fuse because ReLU have multiple consumers')
            return

        # If the fusion is applicable, direct modifications to quantize 1-st and 2-nd inputs
        # are performed. So the data nodes at those inputs shouldn't have more than 1 consumer
        # maximum 2 consumers to the same quantize op (consumed by 1st and 2nd ports).
        # TODO: relax this limitation and duplicate data nodes accordingly to modify the input range freely

        # Provisional limitation that related to binary quantization
        # TODO: Relax it beyond binarization case
        if len(quantize.in_node(1).out_nodes()) != 2 or \
                        len(quantize.in_node(2).out_nodes()) != 2 or \
                        quantize.in_node(1).id != quantize.in_node(2).id or \
                        quantize.levels != 2:
            log.debug('ReluQuantizeFuse: cannot fuse because Quantize op has '
                      'unexpected number of consumers for ports 1 and 2')
            return

        threshold = quantize.in_node(1)

        # As we restricted to binarization case only, so we need to detect from
        # which side of 0 Quantize threshold resides:
        #   if the threshold > 0, it remains the same;
        #   if the threshold == 0, it also remains the same;
        #   if the threshold < 0, it should be modified to -infinity that means that all inputs map to output_high

        modification_mask = threshold.value < 0
        threshold.value[modification_mask] = float('-inf')

        # Remove ReLU as it no longer needed
        remove_op_node_with_data_node(graph, match['relu'])
