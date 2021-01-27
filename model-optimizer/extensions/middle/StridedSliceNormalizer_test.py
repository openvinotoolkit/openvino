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
from mo.ops.strided_slice import StridedSlice
from extensions.middle.StridedSliceNormalizer import StridedSliceNormalizer
from mo.utils.error import Error
from mo.utils.unittest.graph import build_graph, valued_const_with_data, valued_data, regular_op_with_empty_data, \
    connect, shaped_data, shaped_const_with_data, result, build_graph_with_attrs, regular_op
from mo.utils.ir_engine.compare_graphs import compare_graphs

begin = [0, 0, 0]
end = [3, 0, 5]
strides = [1, 1, 1]
input = np.random.rand(10, 10, 10, 10)

# out = input[0:3,..., 0:5]
# out = input[0:3, :, :, 0:5]
# out_shape = (3, 10, 10, 5)

nodes = {
    **valued_const_with_data('input', input),
    **regular_op_with_empty_data('strided_slice', {'op': 'StridedSlice',
                                                   'begin_mask': [1, 1, 1],
                                                   'end_mask': [1, 1, 1],
                                                   'ellipsis_mask': [0, 1, 0],
                                                   'new_axis_mask': [0, 0, 0],
                                                   'shrink_axis_mask': [0, 0, 0],
                                                   'infer': StridedSlice.infer}),
    **regular_op_with_empty_data('strided_slice_normalized', {'op': 'StridedSlice',
                                                   'begin_mask': [0, 1, 1, 0],
                                                   'end_mask': [0, 1, 1, 0],
                                                   'ellipsis_mask': [0, 0, 0, 0],
                                                   'new_axis_mask': [0, 0, 0, 0],
                                                   'shrink_axis_mask': [0, 0, 0, 0],
                                                   'infer': StridedSlice.infer}),
    **valued_const_with_data('begin', int64_array(begin)),
    **valued_const_with_data('end', int64_array(end)),
    **valued_const_with_data('strides', int64_array(strides)),
    **regular_op('res', {'kind': 'op', 'type': 'Result', 'op': 'Result', 'infer': lambda x: None})
}

edges = (
    *connect('input', 'strided_slice:0'),
    *connect('begin', 'strided_slice:1'),
    *connect('end', 'strided_slice:2'),
    *connect('strides', 'strided_slice:3'),
    *connect('strided_slice', 'res')
)

edges_ref = (
    *connect('input', 'strided_slice_normalized:0'),
    *connect('begin', 'strided_slice_normalized:1'),
    *connect('end', 'strided_slice_normalized:2'),
    *connect('strides', 'strided_slice_normalized:3'),
    *connect('strided_slice_normalized', 'res')
)


class TestStridedSliceNormalizer(unittest.TestCase):

    def test_strided_slice_normalizer(self):
        graph = build_graph(nodes, edges, nodes_with_edges_only=True)
        graph_ref = build_graph(nodes, edges_ref, nodes_with_edges_only=True)
        graph.stage = 'middle'
        graph.graph['layout'] = 'NHWC'

        StridedSliceNormalizer().find_and_replace_pattern(graph)
        (flag, resp) = compare_graphs(graph, graph_ref, 'res', check_op_attrs=True)

        pass
