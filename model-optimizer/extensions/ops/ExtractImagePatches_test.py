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
from generator import generator, generate

from extensions.ops.ExtractImagePatches import ExtractImagePatches
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes = {
    'input': {'op': 'Parameter', 'kind': 'op', 'shape': None},
    'input_data': {'value': None, 'kind': 'data', 'shape': None},
    'EIP': {'op': 'ExtractImagePatches', 'kind': 'op', 'sizes': None, 'strides': None, 'rates': None, 'auto_pad': None},
    'EIP_data': {'value': None, 'kind': 'data', 'shape': None},
    'output': {'op': 'Result', 'kind': 'op', 'shape': None},
}

edges = [
    ('input', 'input_data'),
    ('input_data', 'EIP'),
    ('EIP', 'EIP_data'),
    ('EIP_data', 'output'),
]

@generator
class TestExtractImagePatchesPartialInfer(unittest.TestCase):
    @generate(*[
        ([1, 10, 10, 3], [1, 3, 3, 1], [1, 5, 5, 1], [1, 1, 1, 1], 'valid', 'NHWC', [1, 2, 2, 27]),
        ([1, 10, 10, 3], [1, 3, 3, 1], [1, 5, 5, 1], [1, 2, 2, 1], 'valid', 'NHWC', [1, 2, 2, 27]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 8, 8, 1], [1, 1, 1, 1], 'valid', 'NHWC', [1, 1, 1, 48]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 8, 8, 1], [1, 1, 1, 1], 'same_upper', 'NHWC', [1, 2, 2, 48]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 9, 9, 1], [1, 1, 1, 1], 'same_upper', 'NHWC', [1, 2, 2, 48]),
        ([1, 10, 10, 3], [1, 4, 4, 1], [1, 9, 9, 1], [1, 1, 1, 1], 'same_lower', 'NHWC', [1, 2, 2, 48]),
        ([1, 64, 64, 3], [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'valid', 'NHWC', [1, 62, 62, 27]),
        ([1, 64, 64, 3], [1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'same_upper', 'NHWC', [1, 64, 64, 27]),

        ([1, 3, 10, 10], [1, 1, 3, 3], [1, 1, 5, 5], [1, 1, 1, 1], 'valid', 'NCHW', [1, 27, 2, 2]),
        ([1, 3, 10, 10], [1, 1, 4, 4], [1, 1, 8, 8], [1, 1, 1, 1], 'valid', 'NCHW', [1, 48, 1, 1]),

        ([1, 3, 10, 10], [1, 1, 4, 4], [1, 1, 9, 9], [1, 1, 1, 1], 'same_upper', 'NCHW', [1, 48, 2, 2]),
        ([1, 3, 10, 10], [1, 1, 4, 4], [1, 1, 9, 9], [1, 1, 1, 1], 'same_lower', 'NCHW', [1, 48, 2, 2]),

    ])


    def test_eip_infer(self, input_shape, sizes, strides, rates, auto_pad, layout, output_shape):
        graph = build_graph(
            nodes_attrs=nodes,
            edges=edges,
            update_attributes={
                'input': {'shape': int64_array(input_shape)},
                'input_data': {'shape': int64_array(input_shape)},
                'EIP': {'sizes': int64_array(sizes), 'strides': int64_array(strides), 'rates': int64_array(rates),
                        'auto_pad': auto_pad},
            }
        )

        graph.graph['layout'] = layout

        eip_node = Node(graph, 'EIP')
        ExtractImagePatches.infer(eip_node)

        self.assertTrue(np.array_equal(eip_node.out_port(0).data.get_shape(), output_shape))
