# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.ops.upsample import UpsampleOp
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'type': 'Identity', 'kind': 'op'},
                    'upsample': {'type': 'Upsample', 'kind': 'op'},
                    'node_3': {'type': 'Identity', 'kind': 'op'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


class TestUpsampleOp():
    @pytest.mark.parametrize("scales, input_shape, expected_shape",[
        (np.array([1., 1., 2., 2.]), shape_array([1, 3, 227, 227]), shape_array([1, 3, 454, 454])),
        (np.array([1., 1., 2.5, 1.5]), shape_array([1, 5, 227, 227]), shape_array([1, 5, 567, 340])),
        (np.array([1., 1., 1.3, 0.7]), shape_array([1, 14, 1023, 713]), shape_array([1, 14, 1329, 499])),
        (np.array([1., 1., 1.3, 0.7]), shape_array([1, 14, dynamic_dimension_value, 713]),
         shape_array([1, 14, dynamic_dimension_value, 499])),
        (np.array([1., 1., 1.3, 0.7]), shape_array([1, 14, 1023, dynamic_dimension_value]),
         shape_array([1, 14, 1329, dynamic_dimension_value])),
    ])
    def test_upsample_with_scales_infer(self, scales, input_shape, expected_shape):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'upsample'),
                             ('upsample', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1': {'shape': input_shape},
                             'upsample': {'mode': 'linear',
                                          'height_scale': scales[2],
                                          'width_scale': scales[3]}
                             })

        graph.graph['layout'] = 'NCHW'
        upsample_node = Node(graph, 'upsample')
        UpsampleOp.upsample_infer(upsample_node)
        res_shape = graph.node['node_3']['shape']
        assert strict_compare_tensors(expected_shape, res_shape)

    @pytest.mark.parametrize("scales, input_shape, expected_shape",[
        (np.array([1., 1., 2., 2.]), shape_array([1, 3, 227, 227]), shape_array([1, 3, 454, 454])),
        (np.array([1., 1., 2.5, 1.5]), shape_array([1, 5, 227, 227]), shape_array([1, 5, 567, 340])),
        (np.array([1., 1., 1.3, 0.7]), shape_array([1, 14, 1023, 713]), shape_array([1, 14, 1329, 499])),
        (np.array([1., 1., 1.3, 0.7]), shape_array([1, 14, dynamic_dimension_value, 713]),
         shape_array([1, 14, dynamic_dimension_value, 499])),
        (np.array([1., 1., 1.3, 0.7]), shape_array([1, 14, 1023, dynamic_dimension_value]),
         shape_array([1, 14, 1329, dynamic_dimension_value])),
    ])
    def test_upsample_with_second_input_infer(self, scales, input_shape, expected_shape):
        nodes_attributes['scales'] = {'kind': 'data', 'value': scales}
        graph = build_graph(nodes_attributes,
                            [('node_1', 'upsample'),
                             ('scales', 'upsample'),
                             ('upsample', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1': {'shape': input_shape},
                             'upsample': {'mode': 'linear',
                                          'height_scale': None,
                                          'width_scale': None}
                             })

        graph.graph['layout'] = 'NCHW'
        upsample_node = Node(graph, 'upsample')
        UpsampleOp.upsample_infer(upsample_node)
        res_shape = graph.node['node_3']['shape']
        assert strict_compare_tensors(expected_shape, res_shape)
