# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest

from openvino.tools.mo.front.common.partial_infer.eltwise import eltwise_infer, eltwise_reverse_infer
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, strict_compare_tensors, \
    dynamic_dimension_value, reverse_bypass_infer
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.middle.passes.infer import partial_infer
from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.utils.error import Error
from unit_tests.utils.graph import result, regular_op_with_empty_data, connect, \
    build_graph, shaped_parameter

nodes_attributes = {'node_1': {'value': 2, 'kind': 'data'},
                    'node_2': {'value': 3, 'kind': 'data'},
                    'eltw_1': {'kind': 'op'},
                    'node_3': {'value': None, 'kind': 'data'},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    }



class TestEltwiseInfer():
    @pytest.mark.parametrize("value1, shape1, value2, shape2, shape_infer, exp_value, exp_shape",[
        (np.array(2), [], np.array(3), [], lambda a, b: np.multiply(a, b), np.array(6), []),
        (np.array(2), [], np.array(3), [], lambda a, b: np.maximum(a, b), np.array(3), []),
        (np.array(2), [], np.array(3), [], lambda a, b: np.add(a, b), np.array(5), []),
        (None, [1, 5], None, [1, 1], lambda a, b: np.add(a, b), None, [1, 5]),
        (None, [dynamic_dimension_value, 3], None, [1, 1], lambda a, b: np.add(a, b), None,
         [dynamic_dimension_value, 3]),
        (None, [dynamic_dimension_value, 3], None, [1, dynamic_dimension_value], lambda a, b: np.add(a, b), None,
         [dynamic_dimension_value, 3]),
        (None, [4, 5, dynamic_dimension_value, 3], None, [1, dynamic_dimension_value], lambda a, b: np.add(a, b), None,
         [4, 5, dynamic_dimension_value, 3]),
        (None, [1, 10, 20, 30], None, [dynamic_dimension_value, 10, 20, 30], lambda a, b: np.add(a, b), None,
         [dynamic_dimension_value, 10, 20, 30]),
        # dynamic value propagation
        (shape_array([dynamic_dimension_value, 5]), [2], np.array(3), [], lambda a, b: np.add(a, b),
         shape_array([dynamic_dimension_value, 8]), [2]),
        (shape_array([dynamic_dimension_value, 5]), [2], np.array([3, 7]), [], lambda a, b: np.add(a, b),
         shape_array([dynamic_dimension_value, 12]), [2]),
    ])
    def test_eltwise_infer(self, value1, shape1, value2, shape2, shape_infer, exp_value, exp_shape):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': shape_array(value1).shape if value1 is not None else shape_array(shape1),
                                        'value': value1},
                             'node_2': {'shape': shape_array(value2).shape if value2 is not None else shape_array(shape2),
                                        'value': value2}
                             })

        graph.graph['layout'] = 'NCHW'

        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, shape_infer)
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        if exp_value is not None:
            assert strict_compare_tensors(res_value, shape_array(exp_value))
        assert strict_compare_tensors(res_shape, shape_array(exp_shape))

    def test_eltwise_infer_none_val(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 256, 256]), 'value': None},
                             'node_2': {'shape': np.array([1, 3, 256, 256])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        eltwise_infer(eltwise_node, lambda a, b: a * b)
        exp_shape = np.array([1, 3, 256, 256])
        res_shape = graph.node['node_3']['shape']
        res_value = eltwise_node.out_node().value
        for i in range(0, len(exp_shape)):
            assert exp_shape[i] == res_shape[i]

        assert res_value is None

    def test_eltwise_infer_none_min_max(self):
        graph = build_graph(nodes_attributes,
                            [('node_1', 'eltw_1'),
                             ('node_2', 'eltw_1'),
                             ('eltw_1', 'node_3'),
                             ('node_3', 'op_output')
                             ],
                            {'node_3': {'shape': None},
                             'node_1': {'shape': np.array([1, 3, 257, 256])},
                             'node_2': {'shape': np.array([1, 3, 256, 257])}
                             })
        graph.graph['layout'] = 'NCHW'
        eltwise_node = Node(graph, 'eltw_1')

        with pytest.raises(Error, match='Input shapes mismatch*'):
            eltwise_infer(eltwise_node)


dyn = dynamic_dimension_value


class TestElementwiseReverseInfer(unittest.TestCase):
    @staticmethod
    def build_and_test_reverse_inference(inp_shape_1, inp_shape_2, out_shape, ref_shape, auto_broadcast='numpy'):
        in_port_with_defined_shape = 0 if inp_shape_1 is not None else 1
        defined_shape = shape_array(inp_shape_1 if inp_shape_1 is not None else inp_shape_2)

        nodes = {
            **shaped_parameter('undefined_shape_data', None, {'reverse_infer': Parameter.reverse_infer}),
            **shaped_parameter('data', shape_array(defined_shape), {'reverse_infer': Parameter.reverse_infer}),
            **regular_op_with_empty_data('elementwise', {'op': 'Add', 'type': 'Add',
                                                         'infer': eltwise_infer,
                                                         'reverse_infer': eltwise_reverse_infer,
                                                         'auto_broadcast': auto_broadcast}),
            **result('res'),
        }

        edges = [
            *connect('undefined_shape_data', '{}:elementwise'.format(int(not in_port_with_defined_shape))),
            *connect('data', '{}:elementwise'.format(in_port_with_defined_shape)),
            *connect('elementwise', 'res')
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        Node(graph, 'elementwise').out_port(0).data.set_shape(shape_array(out_shape))
        Node(graph, 'elementwise').in_port(in_port_with_defined_shape).data.set_shape(defined_shape)

        partial_infer(graph)
        actual_shape = Node(graph, 'undefined_shape_data').out_port(0).data.get_shape()
        if ref_shape is None:
            assert actual_shape == ref_shape
        else:
            assert strict_compare_tensors(actual_shape, shape_array(ref_shape))

    def test_reverse_infer_1(self):
        self.build_and_test_reverse_inference(inp_shape_1=[dyn, dyn],
                                              inp_shape_2=None,
                                              out_shape=[dyn, dyn, dyn, dyn],
                                              ref_shape=[dyn, dyn, dyn, dyn])

    def test_reverse_infer_2(self):
        self.build_and_test_reverse_inference(inp_shape_1=None,
                                              inp_shape_2=[dyn, dyn],
                                              out_shape=[dyn, dyn, dyn, dyn],
                                              ref_shape=[dyn, dyn, dyn, dyn])

    def test_reverse_infer_3(self):
        self.build_and_test_reverse_inference(inp_shape_1=[1],
                                              inp_shape_2=None,
                                              out_shape=[dyn, 400, 400, 3],
                                              ref_shape=[dyn, 400, 400, 3])

    def test_reverse_infer_4(self):
        self.build_and_test_reverse_inference(inp_shape_1=[4, 1],
                                              inp_shape_2=None,
                                              out_shape=[dyn, dyn, 4, 3],
                                              ref_shape=[dyn, dyn, dyn, 3])

    def test_reverse_infer_5(self):
        self.build_and_test_reverse_inference(inp_shape_1=[4, 1],
                                              inp_shape_2=None,
                                              out_shape=[dyn, dyn, 4, 1],
                                              ref_shape=[dyn, dyn, dyn, 1])

    def test_reverse_infer_6(self):
        # both output and input has the same rank, cannot deduce other inputs rank
        with self.assertRaisesRegex(Error, "Model Optimizer is unable to deduce input shapes"):
            self.build_and_test_reverse_inference(inp_shape_1=[dyn, dyn, dyn, dyn],
                                                  inp_shape_2=None,
                                                  out_shape=[dyn, dyn, 4, 1],
                                                  ref_shape=None)

    def test_reverse_infer_7(self):
        self.build_and_test_reverse_inference(inp_shape_1=[4, dyn],
                                              inp_shape_2=None,
                                              out_shape=[1, dyn, dyn, 1],
                                              ref_shape=[1, dyn, dyn, 1])

    def test_reverse_infer_8(self):
        with self.assertRaisesRegex(AssertionError, "Shapes of Elementwise node '.*' are not compatible"):
            self.build_and_test_reverse_inference(inp_shape_1=[4, dyn],
                                                  inp_shape_2=None,
                                                  out_shape=[1, dyn, 7, 1],
                                                  ref_shape=None)

    def test_reverse_infer_no_broadcast(self):
        self.build_and_test_reverse_inference(inp_shape_1=[1, 4, dyn, dyn],
                                              inp_shape_2=None,
                                              out_shape=[1, dyn, dyn, 1],
                                              ref_shape=[1, 4, dyn, 1],
                                              auto_broadcast='none')


class TestUnaryElementwiseReverseInfer(unittest.TestCase):
    @staticmethod
    def build_and_test_reverse_inference(out_shape):

        nodes = {
            **shaped_parameter('undefined_shape_data', None, {'reverse_infer': Parameter.reverse_infer}),
            **regular_op_with_empty_data('elementwise',
                                         {'op': 'Sqrt', 'type': 'Sqrt',
                                         'infer': eltwise_infer,
                                         'reverse_infer': lambda node: reverse_bypass_infer(node,in_ports=[0])}),
            **result('res'),
        }

        edges = [
            *connect('undefined_shape_data', '0:elementwise'),
            *connect('elementwise', 'res'),
        ]

        graph = build_graph(nodes, edges)
        graph.stage = 'middle'
        Node(graph, 'elementwise').out_port(0).data.set_shape(shape_array(out_shape))

        partial_infer(graph)
        actual_shape = Node(graph, 'elementwise').in_port(0).data.get_shape()

        # check that out_shape is transferred into only existing in_port(0)
        assert strict_compare_tensors(actual_shape, shape_array(out_shape))

    def test_reverse_infer_1(self):
        self.build_and_test_reverse_inference(out_shape=[dyn, dyn, dyn, dyn])

    def test_reverse_infer_2(self):
        self.build_and_test_reverse_inference(out_shape=[dyn, dyn])

    def test_reverse_infer_3(self):
        self.build_and_test_reverse_inference(out_shape=[1, 100])
