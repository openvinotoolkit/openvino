"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.front.mxnet.extractors.slice_axis import mxnet_slice_axis_infer
from mo.front.mxnet.extractors.slice_axis import slice_axis_ext
from mo.front.mxnet.extractors.utils import AttrDictionary
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph


class TestMXNetSliceAxisExtractorOp(unittest.TestCase):
    def test_extract_slice_axis_layer(self):
        graph = build_graph(
            {'node_1': {'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'slice_axis_node': {'type': 'sigmoid', 'kind': 'op', 'op': 'slice_axis', },
             'node_3': {'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [
                ('node_1', 'slice_axis_node'),
                ('slice_axis_node', 'node_3'),
            ],
            {
                'slice_axis_node': {'symbol_dict': {'attrs': {'axis': 0, 'begin': 10, 'end': 25}}},
            })

        exp_attrs = {
            'type': 'Crop',
            'axis': 0,
            'offset': 10,
            'dim': 25
        }

        slice_axis_node = Node(graph, 'slice_axis_node')
        res = slice_axis_ext(AttrDictionary(slice_axis_node['symbol_dict']['attrs']))

        for key in exp_attrs.keys():
            self.assertEqual(res[key], exp_attrs[key])


class TestMXNetSliceAxisInfer(unittest.TestCase):
    def test_slice_axis_infer_layer(self):
        graph = build_graph(
            {'node_1': {'name': 'data', 'type': 'Identity', 'value': None, 'kind': 'op', 'op': 'Parameter'},
             'slice_axis_node': {'name': 'slice_axis_node', 'type': 'sigmoid', 'value': None,
                                 'kind': 'op', 'op': 'slice_axis', },
             'node_3': {'name': 'node_3', 'type': 'Identity', 'value': None, 'kind': 'op'},
             },
            [
                ('node_1', 'slice_axis_node'),
                ('slice_axis_node', 'node_3'),
            ],
            {
                'node_1': {'shape': np.array([1, 1024, 19, 19])},
                'slice_axis_node': {'axis': 1, 'offset': 10, 'dim': 25},
            })

        slice_axis_node = Node(graph, 'slice_axis_node')
        mxnet_slice_axis_infer(slice_axis_node)
        res_shape = [1, 15, 19, 19]
        for i in range(0, len(graph.node['node_3']['shape'])):
            self.assertEqual(graph.node['node_3']['shape'][i], res_shape[i])
