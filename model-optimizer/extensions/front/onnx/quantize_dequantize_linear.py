# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

import numpy as np

from extensions.ops.fakequantize import FakeQuantize
from mo.front.common.replacement import FrontReplacementSubgraph
from mo.front.subgraph_matcher import SubgraphMatch
from mo.graph.graph import Graph, rename_nodes
from mo.ops.const import Const
from mo.utils.error import Error


class QuantizeDequantizeLinear(FrontReplacementSubgraph):
    """
    Fuses QuantizeLinear and DequantizeLinear nodes into single FakeQuantize.
    Covers cases when the values for zero point and scale are same in both QuantizeLinear and DequantizeLinear.
    """
    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('quantize', dict(op='QuantizeLinear')),
                ('dequantize', dict(op='DequantizeLinear')),
            ],
            edges=[
                ('quantize', 'dequantize', {'in': 0}),
            ]
        )

    def replace_sub_graph(self, graph: Graph, match: [dict, SubgraphMatch]):

        q = match['quantize']
        dq = match['dequantize']

        q_scale = q.in_port(1).get_source().node
        q_zerop = q.in_port(2).get_source().node
        dq_scale = dq.in_port(1).get_source().node
        dq_zerop = dq.in_port(2).get_source().node

        inp_port = q.in_port(0).get_source()
        name = inp_port.node.soft_get('name', inp_port.node.id)

        # only constant as for zero_point/scale supported
        if q_scale.soft_get('type') == 'Const' and dq_scale.soft_get('type') == 'Const' and \
                q_zerop.soft_get('type') == 'Const' and dq_zerop.soft_get('type') == 'Const':

            # only patterns with same scale/zero_point values for Q and DQ are supported
            if q_scale.value == dq_scale.value and q_zerop.value == dq_zerop.value:
                log.debug('Found Q-DQ pattern after {}'.format(name))

                zero_point_type = q_zerop.value.dtype
                # data type affects range of output values: [-128..127] or [0..255]
                if zero_point_type == np.int8:
                    output_min_value = -128.0
                    output_max_value = 127.0
                elif zero_point_type == np.uint8:
                    output_min_value = 0.0
                    output_max_value = 255.0
                else:
                    raise Error('Not supported type {} for zero point value in node {}'.format(
                        zero_point_type, q_zerop.soft_get('name')))
                min_value = q_scale.value * (output_min_value - q_zerop.value)
                max_value = q_scale.value * (output_max_value - q_zerop.value)
                input_min = Const(graph, {'value': np.array(min_value)}).create_node()
                input_max = Const(graph, {'value': np.array(max_value)}).create_node()

                FQ = FakeQuantize(graph, {
                    'levels': 256,
                    'name': match['quantize'].name + '_Dequantize/FakeQuantize'
                }).create_node()

                FQ.in_port(0).connect(match['quantize'].in_port(0).get_source())
                FQ.in_port(1).connect(input_min.out_port(0))
                FQ.in_port(2).connect(input_max.out_port(0))
                FQ.in_port(3).connect(input_min.out_port(0))
                FQ.in_port(4).connect(input_max.out_port(0))

                match['dequantize'].out_port(0).get_connection().set_source(FQ.out_port(0))
                dq_name = match['dequantize'].soft_get('name', match['dequantize'].id)
                rename_nodes([(match['dequantize'], dq_name + '/to_be_removed'), (FQ, dq_name)])
            else:
                raise Error('QuantizeLinear and DequantizeLinear (after {}) have different scale or zero-point values, '
                            'cannot fuse into FakeQuantize!'.format(name))
