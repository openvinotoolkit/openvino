# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.Complex import Complex
from unit_tests.utils.graph import build_graph

graph_node_attrs_sizes = {
    'input_real': {'type': 'Parameter', 'kind': 'op'},
    'input_imag': {'type': 'Parameter', 'kind': 'op'},
    'input_real_data': {'kind': 'data', 'shape': None, 'value': None},
    'input_imag_data': {'kind': 'data', 'shape': None, 'value': None},
    'complex': {'op': 'Complex', 'kind': 'op'},
    'complex_data': {'kind': 'data', 'shape': None, 'value': None},
    'op_output': {'kind': 'op', 'op': 'Result'},
}

graph_edges_sizes = [
    ('input_real', 'input_real_data'),
    ('input_imag', 'input_imag_data'),
    ('input_real_data', 'complex', {'in': 0}),
    ('input_imag_data', 'complex', {'in': 1}),
    ('complex', 'complex_data'),
    ('complex_data', 'op_output'),
]


class TestComplexOp():
    @pytest.mark.parametrize("input_shape, output_shape",[
        ([1, 260, 100, 150], [1, 260, 100, 150, 2]),
        ([1, 260, 100], [1, 260, 100, 2]),
        ([5, 14, 300, 40], [5, 14, 300, 40, 2]),
        ([1, 3, 260, 100, 150], [1, 3, 260, 100, 150, 2]),
        ([5, 14, 1000, 300, 40], [5, 14, 1000, 300, 40, 2])
    ])
    def test_complex_op_shape_inference(self, input_shape, output_shape):
        graph = build_graph(nodes_attrs=graph_node_attrs_sizes,
                            edges=graph_edges_sizes,
                            update_attributes={
                                'input_real_data': {'shape': int64_array(input_shape)},
                                'input_imag_data': {'shape': int64_array(input_shape)},
                            })
        node = Node(graph, 'complex')
        Complex.infer(node)

        msg = "Complex operation infer failed for case: expected_shape={}, actual_shape={}"

        assert np.array_equal(graph.node['complex_data']['shape'], int64_array(output_shape)),\
                        msg.format(output_shape, graph.node['complex_data']['shape'])
