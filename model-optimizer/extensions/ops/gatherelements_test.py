"""
 Copyright (C) 2017-2020 Intel Corporation

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

from extensions.ops.gatherelements import GatherElements
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph, regular_op_with_empty_data, result, connect, \
    valued_const_with_data


@generator
class GatherElementsInferTest(unittest.TestCase):
    @generate(*[
        ([[1, 2],
          [3, 4]],
         [[0, 1],
          [0, 0]],
         0,  # axis
         [[1, 4],  # ref_res
          [1, 2]]),

        ([[1, 2],
          [3, 4]],
         [[0, 1],
          [0, 0]],
         1,  # axis
         [[1, 2],  # ref_res
          [3, 3]]),

        ([[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]],
         [[1, 2, 0],
          [2, 0, 0]],
         0,  # axis
         [[4, 8, 3],  # ref_res
          [7, 2, 3]]),

        ([[1, 2],
          [3, 4]],
         [[0, 1],
          [0, 0]],
         -1,  # axis
         [[1, 2],  # ref_res
          [3, 3]]),

        ([  # 3D case
          [[1, 2],
           [3, 4]],
          [[5, 6],
           [7, 8]],
          [[9, 10],
           [11, 12]]
        ],
         [
          [[1, 0],
           [0, 1]],
          [[1, 1],
           [1, 0]],
          [[0, 0],
           [1, 1]]
         ],
         -1,  # axis
         [
          [[2, 1],
           [3, 4]],
          [[6, 6],
           [8, 7]],
          [[9, 9],
           [12, 12]]
         ]),
    ])

    def test_gatherelements_value_infer(self, data, indices, axis, ref_res):
        nodes = {
            **valued_const_with_data('data', int64_array(data)),
            **valued_const_with_data('indices', int64_array(indices)),
            **regular_op_with_empty_data('gather_elements', {'op': 'GatherElements', 'axis': axis}),
            **result()
        }

        graph = build_graph(nodes_attrs=nodes, edges=[
            *connect('data', '0:gather_elements'),
            *connect('indices', '1:gather_elements'),
            *connect('gather_elements', 'output')
        ], nodes_with_edges_only=True)
        graph.stage = 'middle'

        gather_el_node = Node(graph, 'gather_elements')
        GatherElements.infer(gather_el_node)

        res_output_shape = gather_el_node.out_node().shape
        self.assertTrue(np.array_equal(int64_array(ref_res).shape, res_output_shape))

        res_output_value = gather_el_node.out_node().value
        if res_output_value is not None:
            self.assertTrue(np.array_equal(int64_array(ref_res), res_output_value))
