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

from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph


class QuantizeDequantizeRedundant2(FrontReplacementSubgraph):
    """
    Fuses duplicated QuantizeLinear and DequantizeLinear nodes
    (redundancy in the official NV's int8 MLPerf BERT model)
    Covers cases when the values for zero point and scale are same in both QuantizeLinear and DequantizeLinear.
    """
    enabled = True

    def run_before(self):
        from extensions.front.onnx.quantize_dequantize_linear import QuantizeDequantizeLinear
        return [QuantizeDequantizeLinear]

    def pattern(self):
        return dict(
            nodes=[
                ('inp', dict(op='Add')),
                ('quantize0', dict(op='QuantizeLinear')),
                ('dequantize0', dict(op='DequantizeLinear')),
                ('quantize1', dict(op='QuantizeLinear')),
                ('dequantize1', dict(op='DequantizeLinear')),
            ],
            edges=[
                ('inp', 'quantize0', {'in': 0}),
                ('inp', 'quantize1', {'in': 0}),
                ('quantize0', 'dequantize0', {'in': 0}),
                ('quantize1', 'dequantize1', {'in': 0}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):

        q0 = match['quantize0']
        q1 = match['quantize1']

        q0_scale = q0.in_port(1).get_source().node
        q0_zerop = q0.in_port(2).get_source().node
        q1_scale = q1.in_port(1).get_source().node
        q1_zerop = q1.in_port(2).get_source().node

        inp_port = q0.in_port(0).get_source()
        name = inp_port.node.soft_get('name', inp_port.node.id)

        # only constant as for zero_point/scale supported
        if q0_scale.soft_get('type') == 'Const' and q1_scale.soft_get('type') == 'Const' and \
                q0_zerop.soft_get('type') == 'Const' and q1_zerop.soft_get('type') == 'Const':

            # only patterns with same scale/zero_point values for Q and DQ are supported
            if q0_scale.value == q1_scale.value and q0_zerop.value == q1_zerop.value:
                log.debug('Redundant 2Q-DQ pattern after {}'.format(name))

                dests = match['dequantize1'].out_port(0).get_destinations()
                for dest in dests:
                    dest.disconnect()
                    dest.connect(match['dequantize0'].out_port(0))
                graph.remove_nodes_from([match['quantize1'].id, match['dequantize1'].id])
            else:
                log.error('QuantizeLinears in the fan-out have different scale or zero-point values, '
                          'cannot removed!'.format(name))


class QuantizeDequantizeRedundant4(FrontReplacementSubgraph):
    """
    Fuses duplicated QuantizeLinear and DequantizeLinear nodes
    (redundancy in the official NV's int8 MLPerf BERT model)
    Covers cases when the values for zero point and scale are same in both QuantizeLinear and DequantizeLinear.
    """
    enabled = True

    def run_before(self):
        return [QuantizeDequantizeRedundant2]

    def pattern(self):
        return dict(
            nodes=[
                ('inp', dict(op='Add')),
                ('quantize0', dict(op='QuantizeLinear')),
                ('dequantize0', dict(op='DequantizeLinear')),
                ('quantize1', dict(op='QuantizeLinear')),
                ('dequantize1', dict(op='DequantizeLinear')),
                ('quantize2', dict(op='QuantizeLinear')),
                ('dequantize2', dict(op='DequantizeLinear')),
                ('quantize3', dict(op='QuantizeLinear')),
                ('dequantize3', dict(op='DequantizeLinear')),
            ],
            edges=[
                ('inp', 'quantize0', {'in': 0}),
                ('inp', 'quantize1', {'in': 0}),
                ('inp', 'quantize2', {'in': 0}),
                ('inp', 'quantize3', {'in': 0}),
                ('quantize0', 'dequantize0', {'in': 0}),
                ('quantize1', 'dequantize1', {'in': 0}),
                ('quantize2', 'dequantize2', {'in': 0}),
                ('quantize3', 'dequantize3', {'in': 0}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):

        q0 = match['quantize0']
        q1 = match['quantize1']
        q2 = match['quantize2']
        q3 = match['quantize3']

        q0_scale = q0.in_port(1).get_source().node
        q0_zerop = q0.in_port(2).get_source().node
        q1_scale = q1.in_port(1).get_source().node
        q1_zerop = q1.in_port(2).get_source().node
        q2_scale = q2.in_port(1).get_source().node
        q2_zerop = q2.in_port(2).get_source().node
        q3_scale = q3.in_port(1).get_source().node
        q3_zerop = q3.in_port(2).get_source().node

        inp_port = q0.in_port(0).get_source()
        name = inp_port.node.soft_get('name', inp_port.node.id)

        # only constant as for zero_point/scale supported
        if q0_scale.soft_get('type') == 'Const' and q1_scale.soft_get('type') == 'Const' and \
                q0_zerop.soft_get('type') == 'Const' and q1_zerop.soft_get('type') == 'Const' and \
                q2_zerop.soft_get('type') == 'Const' and q2_zerop.soft_get('type') == 'Const' and \
                q3_zerop.soft_get('type') == 'Const' and q3_zerop.soft_get('type') == 'Const':

            # only patterns with same scale/zero_point values for Q and DQ are supported
            if q0_scale.value == q1_scale.value and q0_zerop.value == q1_zerop.value and \
                    q0_scale.value == q2_scale.value and q0_zerop.value == q2_zerop.value and \
                    q0_scale.value == q3_scale.value and q0_zerop.value == q3_zerop.value:
                log.debug('Redundant 4Q-DQ pattern after {}'.format(name))

                dests = match['dequantize1'].out_port(0).get_destinations()
                for dest in dests:
                    dest.disconnect()
                    dest.connect(match['dequantize0'].out_port(0))
                graph.remove_nodes_from([match['quantize1'].id, match['dequantize1'].id])

                dests = match['dequantize2'].out_port(0).get_destinations()
                for dest in dests:
                    dest.disconnect()
                    dest.connect(match['dequantize0'].out_port(0))
                graph.remove_nodes_from([match['quantize2'].id, match['dequantize2'].id])

                dests = match['dequantize3'].out_port(0).get_destinations()
                for dest in dests:
                    dest.disconnect()
                    dest.connect(match['dequantize0'].out_port(0))
                graph.remove_nodes_from([match['quantize3'].id, match['dequantize3'].id])

            else:
                log.error('QuantizeLinears in the fan-out have different scale or zero-point values, '
                          'cannot removed!'.format(name))
