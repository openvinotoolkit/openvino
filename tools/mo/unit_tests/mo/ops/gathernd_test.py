# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.gathernd import GatherND
from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value, strict_compare_tensors
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'data': {'kind': 'op'},
                    'data_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'indices': {'kind': 'op'},
                    'indices_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'gathernd_node': {'op': 'GatherNDUpdate', 'kind': 'op', 'batch_dims': 0, 'version': 'opset8'},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges = [('data', 'data_data', {'in': 0}),
         ('indices', 'indices_data', {'in': 1}),
         ('data_data', 'gathernd_node', {'in': 0}),
         ('indices_data', 'gathernd_node', {'in': 1}),
         ('gathernd_node', 'output', {'out': 0})]

# test data for partial infer: gather elements
inputs = {'data_data': {'shape': int64_array([10, 40]), 'value': None},
           'indices_data': {'shape': int64_array([3, 2]), 'value': None}}

# test data for partial infer: gather slices
inputs1 = {'data_data': {'shape': int64_array([10, 40, 30]), 'value': None},
           'indices_data': {'shape': int64_array([3, 2]), 'value': None}}

# test data for partial infer: gather slices and batch_dims=2
inputs2 = {'data_data': {'shape': int64_array([10, 40, 4, 9]), 'value': None},
           'indices_data': {'shape': int64_array([10, 40, 3, 5, 1]), 'value': None}}

# test data for partial infer: gather slices and batch_dims=3 and indices.shape[-1]=len(data.shape)-batch_dims
inputs3 = {'data_data': {'shape': int64_array([1, 64, 64, 320]), 'value': None},
           'indices_data': {'shape': int64_array([1, 64, 64, 1, 1]), 'value': None}}

# test data for constant folding: gather elements, batch_dims = 0
inputs4 = {'data_data': {'shape': int64_array([2, 2]), 'value': int64_array([[1, 2],
                                                                             [3, 4]])},
           'indices_data': {'shape': int64_array([2, 2]), 'value': int64_array([[0, 0],
                                                                                [1, 0]])}}
output4 = int64_array([1, 3])

# test data for constant folding: gather slices, batch_dims = 0
inputs5 = {'data_data': {'shape': int64_array([2, 3, 4]), 'value': int64_array([[[1, 2, 3, 4],
                                                                                 [5, 6, 7, 8],
                                                                                 [9, 10, 11, 12]],
                                                                                [[13, 14, 15, 16],
                                                                                 [17, 18, 19, 20],
                                                                                 [21, 22, 23, 24]]])},
           'indices_data': {'shape': int64_array([3, 2]), 'value': int64_array([[0, 1],
                                                                                [1, 0],
                                                                                [1, 2]])}}
output5 = int64_array([[5, 6, 7, 8],
                       [13, 14, 15, 16],
                       [21, 22, 23, 24]])

# test data for constant folding: gather slices, batch_dims = 1
inputs6 = {'data_data': {'shape': int64_array([2, 3, 4]), 'value': int64_array([[[1, 2, 3, 4],
                                                                                 [5, 6, 7, 8],
                                                                                 [9, 10, 11, 12]],
                                                                                [[13, 14, 15, 16],
                                                                                 [17, 18, 19, 20],
                                                                                 [21, 22, 23, 24]]])},
           'indices_data': {'shape': int64_array([2, 1]), 'value': int64_array([[1],
                                                                                [0]])}}
output6 = int64_array([[5, 6, 7, 8],
                       [13, 14, 15, 16]])

# test data for constant folding: gather slices with leading dimensions, batch_dims = 2
inputs7 = {'data_data': {'shape': int64_array([2, 3, 4]), 'value': int64_array([[[1, 2, 3, 4],
                                                                                 [5, 6, 7, 8],
                                                                                 [9, 10, 11, 12]],
                                                                                [[13, 14, 15, 16],
                                                                                 [17, 18, 19, 20],
                                                                                 [21, 22, 23, 24]]])},
           'indices_data': {'shape': int64_array([2, 3, 1, 1]), 'value': int64_array([[[[1]],
                                                                                       [[0]],
                                                                                       [[2]]],
                                                                                      [[[0]],
                                                                                       [[2]],
                                                                                       [[2]]]])}}
output7 = int64_array([[2], [5], [11], [13], [19], [23]])

# test data for constant folding: gather elements, batch_dims = 2
inputs8 = {'data_data': {'shape': int64_array([2, 3, 4, 2]),
                         'value': int64_array([[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                                [[9, 10], [11, 12], [13, 14], [15, 16]],
                                                [[17, 18], [19, 20], [21, 22], [23, 24]]],
                                               [[[25, 26], [27, 28], [29, 30], [31, 32]],
                                                [[33, 34], [35, 36], [37, 38], [39, 40]],
                                                [[41, 42], [43, 44], [45, 46], [47, 48]]]])},
           'indices_data': {'shape': int64_array([2, 3, 3, 2]),
                            'value': int64_array([[[[1, 0], [3, 1], [2, 1]],
                                                   [[0, 1], [1, 1], [2, 0]],
                                                   [[3, 0], [3, 1], [2, 1]]],
                                                  [[[2, 0], [1, 1], [3, 1]],
                                                   [[1, 1], [2, 0], [2, 0]],
                                                   [[0, 0], [3, 1], [3, 1]]]])}}
output8 = int64_array([[3, 8, 6],
                       [10, 12, 13],
                       [23, 24, 22],
                       [29, 28, 32],
                       [36, 37, 37],
                       [41, 48, 48]])

# test data for partial infer: gather slices and batch_dims=2
inputs9 = {'data_data': {'shape': shape_array([dynamic_dimension_value, 40, 4, 9]), 'value': None},
           'indices_data': {'shape': shape_array([dynamic_dimension_value, 40, 3, 5, 1]), 'value': None}}

# test data for partial infer: gather slices and batch_dims=2
inputs10 = {'data_data': {'shape': shape_array([40, dynamic_dimension_value, 4, 9]), 'value': None},
           'indices_data': {'shape': shape_array([40, dynamic_dimension_value, 3, 5, 1]), 'value': None}}

# test data for partial infer: gather slices and batch_dims=2
inputs11 = {'data_data': {'shape': shape_array([dynamic_dimension_value, 40, 4, 9]), 'value': None},
           'indices_data': {'shape': shape_array([40, dynamic_dimension_value, 3, 5, 1]), 'value': None}}

# invalid test case with incorrect rank for indices
inputs_inv1 = {'data_data': {'shape': int64_array([10, 40]), 'value': None},
               'indices_data': {'shape': int64_array([5, 3, 4]), 'value': None}}

# invalid test case with unequal batch dimensions, batch_dims = 2
inputs_inv2 = {'data_data': {'shape': int64_array([10, 40, 20]), 'value': None},
               'indices_data': {'shape': int64_array([5, 3, 4]), 'value': None}}

# invalid test case with indices rank greater than a rank of data excluding batch dimensions, batch_dims = 2
inputs_inv3 = {'data_data': {'shape': int64_array([10, 40, 20, 10, 2]), 'value': None},
               'indices_data': {'shape': int64_array([10, 40, 4]), 'value': None}}


class TestGatherND_5(unittest.TestCase):
    def setUp(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 0
        nodes_attributes['gathernd_node']['version'] = 'opset5'

    def test_partial_infer_gather_element(self):
        graph = build_graph(nodes_attributes, edges, inputs)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = int64_array([3])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice(self):
        graph = build_graph(nodes_attributes, edges, inputs1)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = int64_array([3, 30])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs2)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = int64_array([400, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims3(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 3
        graph = build_graph(nodes_attributes, edges, inputs3)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = int64_array([4096, 1])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2_dynamic1(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs9)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = shape_array([dynamic_dimension_value, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(strict_compare_tensors(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2_dynamic2(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs10)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = shape_array([dynamic_dimension_value, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(strict_compare_tensors(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2_dynamic3(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs11)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = shape_array([dynamic_dimension_value, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(strict_compare_tensors(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer4(self):
        graph = build_graph(nodes_attributes, edges, inputs4)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output4, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output4, res_output_value))

    def test_infer5(self):
        graph = build_graph(nodes_attributes, edges, inputs5)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output5, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output5, res_output_value))

    def test_infer6(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 1
        graph = build_graph(nodes_attributes, edges, inputs6)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output6, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output6, res_output_value))

    def test_infer7(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs7)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        output = output7.reshape([6, 1])
        self.assertTrue(np.array_equal(output, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output, res_output_value))

    def test_infer8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs8)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output8, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output8, res_output_value))

    def test_infer9(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs8)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        self.assertTrue(np.array_equal(output8, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output8, res_output_value))

    def test_infer9_opset_5(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs8)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        output = output8.reshape([6, 3])
        self.assertTrue(np.array_equal(output, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output, res_output_value))

    def test_infer_invalid1(self):
        graph = build_graph(nodes_attributes, edges, inputs_inv1)
        gathernd_node = Node(graph, 'gathernd_node')
        self.assertRaises(AssertionError, GatherND.infer, gathernd_node)

    def test_infer_invalid2(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs_inv2)
        gathernd_node = Node(graph, 'gathernd_node')
        self.assertRaises(AssertionError, GatherND.infer, gathernd_node)

    def test_infer_invalid3(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        graph = build_graph(nodes_attributes, edges, inputs_inv3)
        gathernd_node = Node(graph, 'gathernd_node')
        self.assertRaises(AssertionError, GatherND.infer, gathernd_node)


    def test_partial_infer_gather_slice_batch_dims2_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs2)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = int64_array([10, 40, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims3_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 3
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs3)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = int64_array([1, 64, 64, 1])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2_dynamic1_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs9)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = shape_array([dynamic_dimension_value, 40, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(strict_compare_tensors(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2_dynamic2_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs10)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = shape_array([40, dynamic_dimension_value, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(strict_compare_tensors(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_partial_infer_gather_slice_batch_dims2_dynamic3_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs11)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # prepare reference results
        ref_output_shape = shape_array([40, 40, 3, 5, 9])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(strict_compare_tensors(ref_output_shape, res_output_shape),
                        'values do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer7_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs7)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        output = output7.reshape([2, 3, 1])

        self.assertTrue(np.array_equal(output, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output, res_output_value))

    def test_infer8_opset8(self):
        nodes_attributes['gathernd_node']['batch_dims'] = 2
        nodes_attributes['gathernd_node']['version'] = 'opset8'
        graph = build_graph(nodes_attributes, edges, inputs8)
        gathernd_node = Node(graph, 'gathernd_node')
        GatherND.infer(gathernd_node)

        # get the result
        res_output_value = graph.node['output']['value']

        output = output8.reshape([2, 3, 3])

        self.assertTrue(np.array_equal(output, res_output_value),
                        'values do not match expected: {} and given: {}'.format(output, res_output_value))
