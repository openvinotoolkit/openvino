# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.reshape import Reshape
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'input': {
        'kind': 'op',
        'op': 'Parameter',
        'shape': None,
        'value': None,
    },
    'data': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'output_shape': {
        'kind': 'op',
        'op': 'Const',
        'value': None,
        'shape': None,
    },
    'output_shape_data': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'reshape': {
        'op': 'Reshape',
        'kind': 'op',
        'special_zero': True,
    },
    'reshape_out': {
        'kind': 'data',
        'shape': None,
        'value': None,
    }
}


class TestReshapeShapeInfer():
    @pytest.mark.parametrize("input_value, input_shape, output_shape, ref_value, ref_shape",[
        (None, shape_array([1, 100, 4]), shape_array([-1, 25]), None, [16, 25]),
        (None, shape_array([5, 100, 4]), shape_array([0, -1, 25]), None, [5, 16, 25]),
        (None, shape_array([5, dynamic_dimension_value, 4]), shape_array([4, -1, 5]), None,
         shape_array([4, dynamic_dimension_value, 5])),
        (None, shape_array([5, dynamic_dimension_value, 4]), shape_array([4, dynamic_dimension_value, 5]), None,
         shape_array([4, dynamic_dimension_value, 5])),
        (None, shape_array([dynamic_dimension_value, 4, 5]), shape_array([0, -1]), None,
         shape_array([dynamic_dimension_value, 20])),
        (None, shape_array([dynamic_dimension_value, 4, 5]), shape_array([5, -1, dynamic_dimension_value]),
         None, shape_array([5, dynamic_dimension_value, dynamic_dimension_value])),
        (None, shape_array([dynamic_dimension_value, 1, 546]), shape_array([dynamic_dimension_value, -1, 91]),
         None, shape_array([dynamic_dimension_value, dynamic_dimension_value, 91])),
        (None, shape_array([5, dynamic_dimension_value, 8]), shape_array([4, -1]),
         None, shape_array([4, dynamic_dimension_value])),
        (None, shape_array([dynamic_dimension_value]), shape_array([5]), None, shape_array([5])),
        (None, shape_array([dynamic_dimension_value]), shape_array([0]), None, shape_array([dynamic_dimension_value])),
        (None, shape_array([dynamic_dimension_value]), shape_array([-1]), None, shape_array([dynamic_dimension_value])),
        (None, shape_array([dynamic_dimension_value]), shape_array([dynamic_dimension_value]), None,
         shape_array([dynamic_dimension_value])),
        # even though the target shape is dynamic since all the inputs are static so we can calculate output
        (None, shape_array([5, 3, 8]), shape_array([4, dynamic_dimension_value]), None, shape_array([4, 30])),
        (None, shape_array([3, 14, 5]), shape_array([dynamic_dimension_value, 2, 0]), None, shape_array([21, 2, 5])),
        (shape_array([1, 2, dynamic_dimension_value, 4, 5, 6]), shape_array([6]), shape_array([-1, 2]),
         shape_array([1, 2, dynamic_dimension_value, 4, 5, 6]).reshape((3, 2)), shape_array([3, 2])),
    ])
    def test_reshape_infer(self, input_value, input_shape, output_shape, ref_value, ref_shape):
        graph = build_graph(nodes_attributes,
                            [('input', 'data'),
                             ('data', 'reshape'),
                             ('output_shape', 'output_shape_data'),
                             ('output_shape_data', 'reshape'),
                             ('reshape', 'reshape_out')],
                            {'data': {'shape': input_shape, 'value': input_value},
                             'output_shape': {'value': output_shape, 'shape': output_shape.shape},
                             'output_shape_data': {'value': output_shape, 'shape': output_shape.shape},
                             })
        node = Node(graph, 'reshape')
        Reshape.infer(node)
        if ref_value is not None:
            assert strict_compare_tensors(node.out_port(0).data.get_value(), shape_array(ref_value))
        assert strict_compare_tensors(node.out_port(0).data.get_shape(), shape_array(ref_shape))
