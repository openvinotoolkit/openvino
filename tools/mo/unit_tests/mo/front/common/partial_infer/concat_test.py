# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

import numpy as np

from openvino.tools.mo.front.common.partial_infer.concat import concat_infer
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import build_graph

nodes_attributes = {'node_1': {'kind': 'data', 'value': None},
                    'node_2': {'kind': 'data', 'value': None},
                    'concat': {'type': 'Concat', 'kind': 'op'},
                    'node_3': {'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }


class TestConcatPartialInfer():
    @pytest.mark.parametrize("shape1, shape2, output_shape, axis",[([1, 3, 227, 227], [1, 3, 220, 227],
                                                                    [1, 3, 447, 227], 2),
                ([1, 3, 227, 227], [1, 3, 227, 220], [1, 3, 227, 447], -1),
                ([1, 3, dynamic_dimension_value, 227], [1, dynamic_dimension_value, 227, 220], [1, 3, 227, 447], -1),
                ([1, 3, 10, 227], [1, 3, 10, dynamic_dimension_value], [1, 3, 10, dynamic_dimension_value], -1),
                ])
    def test_concat_infer(self, shape1, shape2, output_shape, axis):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1': {'shape': shape_array(shape1)},
                             'node_2': {'shape': shape_array(shape2)},
                             'concat': {'axis': axis}
                             })

        concat_node = Node(graph, 'concat')
        concat_infer(concat_node)
        res_shape = graph.node['node_3']['shape']
        assert strict_compare_tensors(output_shape, res_shape)

    @pytest.mark.parametrize("value1, value2, output_value, axis",[(shape_array([1]),
                    shape_array([4]), shape_array([1, 4]), 0),
                (shape_array([dynamic_dimension_value]), shape_array([4]),
                 shape_array([dynamic_dimension_value, 4]), -1),
                ])
    def test_concat_value_infer(self, value1, value2, output_value, axis):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': output_value.shape, 'value': output_value},
                             'node_1': {'shape': value1.shape, 'value': value1},
                             'node_2': {'shape': value2.shape, 'value': value2},
                             'concat': {'axis': axis}
                             })

        concat_node = Node(graph, 'concat')
        concat_infer(concat_node)
        res_value = graph.node['node_3']['value']
        assert strict_compare_tensors(output_value, res_value)

    def test_concat_infer_not_match(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None, 'value': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': np.array([1, 2, 227, 227])},
                             'concat': {'axis': 2}
                             })

        concat_node = Node(graph, 'concat')
        with pytest.raises(Error, match="Concat input shapes do not match for node*"):
            concat_infer(concat_node)

    def test_concat_infer_no_shape(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'concat'),
                             ('node_2', 'concat'),
                             ('concat', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 227, 227])},
                             'node_2': {'shape': None},
                             'concat': {'axis': 2}
                             })

        concat_node = Node(graph, 'concat')
        with pytest.raises(Error, match="One of the input shapes is not defined for node *"):
            concat_infer(concat_node)
