# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.graph.graph import Graph, rename_nodes
from openvino.tools.mo.middle.quantize_linear_resolver import QuantizeLinearResolver
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern


class QuantizeDequantizeLinearResolver(MiddleReplacementPattern):
    """
    This transformation replaces QuantizeLinear in pair QuantizeLinear/DequantizeLinear with
    constant inputs to FakeQuantize with flag stop_value_propagation=True. This transformation prepare FakeQuantize for
    ConvertQuantizeDequantize in offline transformations.
    """
    enabled = True
    graph_condition = [lambda graph: graph.graph['layout'] == 'NCHW']

    def pattern(self):
        return dict(
            nodes=[('const_input', dict(kind='op', op='Const')),
                   ('const_input_d', dict(kind='data')),
                   ('quantize', dict(kind='op', op='QuantizeLinear')),
                   ('quantize_d', dict(kind='data')),
                   ('dequantize', dict(kind='op', op='DequantizeLinear')),
                   ],
            edges=[('const_input', 'const_input_d'),
                   ('const_input_d', 'quantize', {'in': 0}),
                   ('quantize', 'quantize_d'),
                   ('quantize_d', 'dequantize', {'in': 0})
                   ]
        )

    def run_after(self):
        from openvino.tools.mo.middle.quantize_fuses import MarkNodesToFuseUpToFakeQuantize
        return [MarkNodesToFuseUpToFakeQuantize]

    def replace_pattern(self, graph: Graph, match: dict):
        dequantize_node = match['dequantize']
        quantize_node = match['quantize']

        scale_zerop_is_exist = quantize_node.is_in_port_connected(1) and quantize_node.is_in_port_connected(2) and \
                               dequantize_node.is_in_port_connected(1) and dequantize_node.is_in_port_connected(2)
        if not scale_zerop_is_exist:
            return
        q_scale = quantize_node.in_port(1).get_source().node
        q_zerop = quantize_node.in_port(2).get_source().node
        dq_scale = dequantize_node.in_port(1).get_source().node
        dq_zerop = dequantize_node.in_port(2).get_source().node
        scales_and_zerop_is_const = q_scale.soft_get('type') == 'Const' and dq_scale.soft_get('type') == 'Const' and \
                                    q_zerop.soft_get('type') == 'Const' and dq_zerop.soft_get('type') == 'Const'
        scales_and_zerop_equals = np.array_equal(q_scale.value, dq_scale.value) and \
                                  np.array_equal(q_zerop.value, dq_zerop.value)

        # only constant as for zero_point/scale supported
        # only patterns with same scale/zero_point values for Q and DQ are supported
        if not (scales_and_zerop_is_const or scales_and_zerop_equals):
            return

        QuantizeLinearResolver.quantize_to_fakequantize(graph, quantize_node, True)
        quantize_node['isolated'] = True
