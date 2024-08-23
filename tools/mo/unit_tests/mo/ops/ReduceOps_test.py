# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import unittest

import numpy as np

from openvino.tools.mo.ops.ReduceOps import reduce_infer
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, strict_compare_tensors, is_fully_defined
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph, regular_op_with_shaped_data, result, connect, valued_const_with_data

nodes_attributes = {
    **regular_op_with_shaped_data('data', [1, 3, 224, 224], {'type': 'Parameter', 'value': None,
                                                             '_out_port_data_type': {0: np.float32}}),
    **valued_const_with_data('axis', int64_array(0)),
    **regular_op_with_shaped_data('reduce_lp', None, {'op': 'ReduceLp', 'type': None, 'name': 'my_reduce_lp'}),
    **regular_op_with_shaped_data('identity', None, {'op': 'Identity', 'name': 'identity'}),
    **result('output'),
}


class TestReduceLpTest():
    @unittest.skip("Skipped due to function array_equal failure")
    @pytest.mark.parametrize("shape, axes, keepdims, p",[
        ([3, 2, 2], [0], True, 1),
        ([3, 2, 2], [0], True, 2),
        ([3, 2, 2], [1], True, 2),
        ([3, 2, 2], [2], True, 2),
        ([3, 2, 2], [0], False, 1),
        ([3, 2, 2], [0], False, 2),
        ([3, 2, 2], [1], False, 2),
        ([3, 2, 2], [2], False, 2),
    ])
    def test_reduce_lp(self, shape, axes, keepdims, p):
        data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
        reduced = np.power(np.sum(a=np.abs(np.power(data, p)), axis=tuple(axes), keepdims=keepdims), 1 / p)
        axis = int64_array(axes)
        p = int64_array(p)
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:reduce_lp'),
                             *connect('axis', '1:reduce_lp'),
                             *connect('reduce_lp', '0:identity'),
                             ('identity', 'identity_d', {'out': 0}),
                             ('identity_d', 'output')
                             ],
                            {'data_d': {'value': data, 'shape': data.shape},
                             'axis_d': {'value': axis, 'shape': axis.shape},
                             'reduce_lp': {'keep_dims': keepdims}},
                            nodes_with_edges_only=True)

        reduce_node = Node(graph, 'reduce_lp')
        reduce_node.op = reduce_node.type = 'ReduceL' + str(p)
        reduce_infer(reduce_node)
        assert np.array_equal(reduce_node.out_port(0).data.get_value(), reduced)

    @pytest.mark.parametrize("shape, axes, keepdims, p",[
        ([3, 2, 2], [0], True, 1),
        ([3, 2, 2], [2], False, 2),
        ([3, 2, 2], [0, 2], False, 2),
    ])
    def test_reduce_dynamic(self, shape, axes, keepdims, p):
        false_mask = np.zeros(shape)
        false_mask[0][1][1] = True
        data = np.ma.masked_array(np.ones(shape), mask=false_mask)
        assert not is_fully_defined(data)
        reduced_tensor = np.sum(data, axis=tuple(axes), keepdims=keepdims)
        # create an array of all masked elements which is the expected result of the reduce of the tensor with dynamic
        # values
        fully_undefined = np.ma.masked_array(reduced_tensor, mask=np.ones(reduced_tensor.shape))
        axis = int64_array(axes)
        p = int64_array(p)
        graph = build_graph(nodes_attributes,
                            [*connect('data', '0:reduce_lp'),
                             *connect('axis', '1:reduce_lp'),
                             *connect('reduce_lp', '0:identity'),
                             ('identity', 'identity_d', {'out': 0}),
                             ('identity_d', 'output')
                             ],
                            {'data_d': {'value': data, 'shape': data.shape},
                             'axis_d': {'value': axis, 'shape': axis.shape},
                             'reduce_lp': {'keep_dims': keepdims}},
                            nodes_with_edges_only=True)

        reduce_node = Node(graph, 'reduce_lp')
        reduce_node.op = reduce_node.type = 'ReduceL' + str(p)
        reduce_infer(reduce_node)
        assert strict_compare_tensors(reduce_node.out_port(0).data.get_value(), fully_undefined)
