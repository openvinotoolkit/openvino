# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.squeeze import Squeeze
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph

nodes_attributes = {
    'data': {
        'kind': 'data',
        'shape': None,
        'value': None,
    },
    'squeeze_dims': {
        'kind': 'op',
        'op': 'Const',
        'value': np.array([]),
        'shape': None,
    },
    'squeeze_dims_data': {
        'kind': 'data',
        'shape': None,
        'value': np.array([]),
    },
    'squeeze': {
        'op': 'Squeeze',
        'kind': 'op',
    },
    'data_out': {
        'kind': 'data',
        'shape': None,
        'value': None,
    }
}


class TestSqueezeInfer():
    @pytest.mark.parametrize("input_value, input_shape, squeeze_dims, ref_value, ref_shape",[
        (None, shape_array([1, 2, 1, 4]), shape_array([2]), None, [1, 2, 4]),
                # allow squeezing dynamic dimensions
                (None, shape_array([1, 2, dynamic_dimension_value, 4]), shape_array([2]), None, [1, 2, 4]),
                (None, shape_array([1, 2, 1, 4]), shape_array([]), None, [2, 4]),
                (None, shape_array([1, dynamic_dimension_value, 1, 4]), shape_array([]), None,
                 shape_array([dynamic_dimension_value, 4])),
                # do not allow squeeze dimensions not equal to 1
                (None, shape_array([1, 2, 1, 4]), shape_array([1]), None, None),
                # do not allow squeeze input shape to be None
                (None, None, shape_array([1]), None, None),
                ])
    def test_squeeze_squeeze_dims(self, input_value, input_shape, squeeze_dims, ref_value, ref_shape):
        graph = build_graph(nodes_attributes,
                            [('data', 'squeeze'),
                             ('squeeze_dims', 'squeeze_dims_data'),
                             ('squeeze_dims_data', 'squeeze'),
                             ('squeeze', 'data_out')],
                            {'data': {'shape': input_shape, 'value': input_value},
                             'squeeze_dims': {'value': squeeze_dims, 'shape': squeeze_dims.shape},
                             'squeeze_dims_data': {'value': squeeze_dims, 'shape': squeeze_dims.shape},
                             })
        node = Node(graph, 'squeeze')
        if ref_shape is None:  # the test should fail
            with pytest.raises(Error):
                Squeeze.infer(node)
        else:
            Squeeze.infer(node)
            if ref_value is not None:
                assert strict_compare_tensors(node.out_port(0).data.get_value(), ref_value)
            assert strict_compare_tensors(node.out_port(0).data.get_shape(), ref_shape)
