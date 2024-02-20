# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.unsqueeze import Unsqueeze
from openvino.tools.mo.utils.ir_engine.compare_graphs import compare_graphs
from unit_tests.utils.graph import build_graph


class TestUnsqueezeOp():
    nodes_attributes = {
        'data_1': {
            'kind': 'data',
            'shape': None,
            'value': None,
        },
        'unsq': {
            'op': 'Unsqueeze',
            'kind': 'op',
        },
        'unsq_dims_const': {
            'op': 'Const',
            'kind': 'op',
        },
        'unsq_dims': {
            'kind': 'data',
        },
        'data_2': {
            'kind': 'data',
            'shape': None,
            'value': None,
        }
    }

    @pytest.mark.parametrize("input_shape, unsq_dims, output_shape, ref_uns_dims, input_value, output_value",
                [(shape_array([1, 3, 64, 64]), int64_array([0, 4]), shape_array([1, 1, 3, 64, 1, 64]),
                 int64_array([0, 4]), None, None),
                (shape_array([2, 3, 64, 64]), int64_array([-1]), shape_array([2, 3, 64, 64, 1]), int64_array([4]), None,
                 None),
                (shape_array([2, 3, dynamic_dimension_value, 64]), int64_array([0]),
                 shape_array([1, 2, 3, dynamic_dimension_value, 64]), int64_array([0]), None, None),
                (shape_array([1, 2]), int64_array([-1]), shape_array([1, 2, 1]), int64_array([2]),
                 shape_array([5, dynamic_dimension_value]).reshape((1, 2)),
                 shape_array([5, dynamic_dimension_value]).reshape((1, 2, 1))),
                ])
    def test_unsqueeze_infer(self, input_shape, unsq_dims, output_shape, ref_uns_dims, input_value, output_value):
        graph = build_graph(self.nodes_attributes,
                            [('data_1', 'unsq'),
                             ('unsq_dims_const', 'unsq_dims'),
                             ('unsq_dims', 'unsq'),
                             ('unsq', 'data_2')],
                            {'data_1': {'shape': input_shape, 'value': input_value},
                             'unsq_dims': {'value': unsq_dims, 'shape': unsq_dims.shape},
                             'unsq_dims_const': {'value': unsq_dims, 'shape': unsq_dims.shape},
                             })

        graph_ref = build_graph(self.nodes_attributes,
                                [('data_1', 'unsq'),
                                 ('unsq_dims_const', 'unsq_dims'),
                                 ('unsq_dims', 'unsq'),
                                 ('unsq', 'data_2')],
                                {'data_1': {'shape': input_shape, 'value': input_value},
                                 'unsq_dims': {'value': ref_uns_dims, 'shape': ref_uns_dims.shape},
                                 'unsq_dims_const': {'value': ref_uns_dims, 'shape': ref_uns_dims.shape},
                                 'data_2': {'shape': output_shape, 'value': output_value},
                                 })

        unsqueeze_node = Node(graph, 'unsq')
        Unsqueeze.infer(unsqueeze_node)

        (flag, resp) = compare_graphs(graph, graph_ref, 'data_2')
        assert flag, resp
        assert strict_compare_tensors(Node(graph, 'data_2').shape, Node(graph_ref, 'data_2').shape)
        if Node(graph_ref, 'data_2').value is not None:
            assert strict_compare_tensors(Node(graph, 'data_2').value, Node(graph_ref, 'data_2').value)
