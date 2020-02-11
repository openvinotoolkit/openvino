"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import unittest

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.op import PermuteAttrs
from mo.ops.squeeze import Squeeze
from mo.utils.unittest.graph import build_graph

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


class TestSqueezeInfer(unittest.TestCase):
    def test_squeeze_squeeze_dims(self):
        graph = build_graph(nodes_attributes,
                            [('data', 'squeeze'),
                             ('squeeze_dims', 'squeeze_dims_data'),
                             ('squeeze_dims_data', 'squeeze'),
                             ('squeeze', 'data_out')],
                            {'data': {'shape': np.array([1, 2, 1, 4])},
                             'squeeze_dims': {'value': np.array([2]), 'shape': np.array([1])},
                             'squeeze_dims_data': {'value': np.array([2]), 'shape': np.array([1])},
                             })
        node = Node(graph, 'squeeze')
        Squeeze.infer(node)

        self.assertTrue(np.all(node.out_port(0).data.get_shape() == [1, 2, 4]))

    def test_squeeze_empty_squeeze_dims(self):
        graph = build_graph(nodes_attributes,
                            [('data', 'squeeze'),
                             ('squeeze_dims', 'squeeze_dims_data'),
                             ('squeeze_dims_data', 'squeeze'),
                             ('squeeze', 'data_out')],
                            {'data': {'shape': np.array([1, 2, 1, 4])},
                             'squeeze_dims': {'value': np.array([]), 'shape': np.array([1])},
                             'squeeze_dims_data': {'value': np.array([]), 'shape': np.array([1])},
                             })
        node = Node(graph, 'squeeze')
        Squeeze.infer(node)

        self.assertTrue(np.all(node.out_port(0).data.get_shape() == [2, 4]))
