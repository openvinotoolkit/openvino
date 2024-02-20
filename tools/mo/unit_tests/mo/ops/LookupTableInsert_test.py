# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from openvino.tools.mo.ops.LookupTableInsert import LookupTableInsert
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Node
from unit_tests.utils.graph import build_graph

nodes_attributes = {'table': {'kind': 'op'},
                    'table_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'keys': {'kind': 'op'},
                    'keys_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'values': {'kind': 'op'},
                    'values_data': {'shape': None, 'value': None, 'kind': 'data'},
                    'lookuptableinsert_node': {'op': 'LookupTableInsert', 'kind': 'op'},
                    'output': {'shape': None, 'value': None, 'kind': 'data'}}

# graph 1
edges1 = [('table', 'table_data'),
          ('keys', 'keys_data'),
          ('values', 'values_data'),
          ('table_data', 'lookuptableinsert_node', {'in': 0}),
          ('keys_data', 'lookuptableinsert_node', {'in': 1}),
          ('values_data', 'lookuptableinsert_node', {'in': 2}),
          ('lookuptableinsert_node', 'output')]

# valid test case
inputs1 = {'table_data': {},
           'keys_data': {'shape': int64_array([4])},
           'values_data': {'shape': int64_array([4])}}

# invalid test case
inputs2 = {'table_data': {},
           'keys_data': {'shape': int64_array([5, 2])},
           'values_data': {'shape': int64_array([4])}}

class TestLookupTableInsert(unittest.TestCase):
    def test_infer1(self):
        graph = build_graph(nodes_attributes, edges1, inputs1)
        lookuptableinsert_node = Node(graph, 'lookuptableinsert_node')
        LookupTableInsert.infer(lookuptableinsert_node)

        # prepare reference results
        ref_output_shape = int64_array([])

        # get the result
        res_output_shape = graph.node['output']['shape']

        self.assertTrue(np.array_equal(ref_output_shape, res_output_shape),
                        'shapes do not match expected: {} and given: {}'.format(ref_output_shape, res_output_shape))

    def test_infer_invalid1(self):
        graph = build_graph(nodes_attributes, edges1, inputs2)
        lookuptableinsert_node = Node(graph, 'lookuptableinsert_node')
        self.assertRaises(AssertionError, LookupTableInsert.infer, lookuptableinsert_node)
