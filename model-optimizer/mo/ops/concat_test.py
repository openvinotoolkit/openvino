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

from mo.front.common.partial_infer.concat import concat_infer
from mo.ops.concat import Concat
from mo.utils.unittest.graph import build_graph


class TestConcatOp(unittest.TestCase):
    nodes_attributes = {
        'node_1': {
            'shape': np.array([227, 227, 227, 227])
        },
        'concat_node': {
        },
        'node_3': {
            'kind': 'data'
        }
    }

    def test_concat_op(self):
        graph = build_graph(self.nodes_attributes,
                            [
                                ('node_1', 'concat_node'),
                                ('concat_node', 'node_3')
                            ])
        concat_node = Concat(graph, self.nodes_attributes['concat_node']).add_node()
        self.assertEqual(concat_node.type, 'Concat')
        self.assertEqual(concat_node.op, 'Concat')
        self.assertEqual(concat_node.infer, concat_infer)
