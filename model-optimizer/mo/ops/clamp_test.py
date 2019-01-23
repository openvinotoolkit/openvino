"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.ops.clamp import Clamp
from mo.utils.unittest.graph import build_graph


class TestClampOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([227, 227, 227, 227])
        },
        'clamp_node': {
        },
        'node_3': {
            'kind': 'data'
        }
    }

    def test_clamp_op(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'clamp_node'),
                                ('clamp_node', 'node_3')
                            ])
        clamp_node = Clamp(graph, self.nodes_attributes['clamp_node']).add_node()
        self.assertEqual(clamp_node.type, 'Clamp')
        self.assertEqual(clamp_node.op, 'Clamp')
        self.assertEqual(clamp_node.infer, copy_shape_infer)
