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
