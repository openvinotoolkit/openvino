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

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.ops.broadcast import Broadcast
from mo.utils.unittest.graph import build_graph, valued_const_with_data, regular_op_with_empty_data, \
    shaped_data


@generator
class BroadcastTest(unittest.TestCase):
    @generate(*[
        ([1], [3, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None, 'numpy', True),
        ([1], [3, 3], [3, 3], None, 'numpy', False),
        ([1], [2, 2], [[1, 1]], [0], 'explicit', True),
        ([1], [2, 2], [1, 2], [0], 'explicit', False),
    ])
    def test_broadcast(self, data, target_shape, ref_out, axes_mapping=None, mode='numpy', val_broadcast=True):
        input = valued_const_with_data('data', int64_array(data)) if val_broadcast else shaped_data('data', int64_array(data))
        nodes = {
            **input,
            **valued_const_with_data('target_shape', int64_array(target_shape)),
            **regular_op_with_empty_data('broadcast', {'op': 'Broadcast', 'mode': mode}),
        }

        edges = [('data', 'broadcast'),
                 ('target_shape', 'broadcast'),
                 ('broadcast', 'broadcast_d')]

        if axes_mapping is not None:
            nodes.update(**valued_const_with_data('axes_mapping', int64_array(axes_mapping)))
            edges.append(('axes_mapping', 'broadcast'))
        graph = build_graph(nodes, edges)

        broadcast_node = Node(graph, 'broadcast')
        Broadcast.infer(broadcast_node)
        if val_broadcast:
            self.assertTrue(np.array_equal(broadcast_node.out_node().value, np.array(ref_out)))
        else:
            self.assertTrue(np.array_equal(broadcast_node.out_node().shape, np.array(ref_out)))
