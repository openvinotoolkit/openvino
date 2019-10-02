"""
 Copyright (c) 2018-2019 Intel Corporation

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

import logging as log

import numpy as np

from extensions.middle.ConvertGroupedStridedSlice import ConvertGroupedStridedSlice
from extensions.middle.SliceConverter import ConvertSlice
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.middle.passes.eliminate import remove_op_node_with_data_node
from mo.middle.replacement import MiddleReplacementPattern


class UselessStridedSliceEraser(MiddleReplacementPattern):
    enabled = True
    force_shape_inference = True

    def run_before(self):
        return [ConvertGroupedStridedSlice]

    def run_after(self):
        return [ConvertSlice]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('strided_slice', dict(kind='op', op='StridedSlice'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node_ss = match['strided_slice']
        output_data_node = node_ss.out_node(0)
        input_data_node = node_ss.in_node(0)

        out_shape = output_data_node.shape

        if not np.all(node_ss.shrink_axis_mask == 0):
            out_shape = list(out_shape)
            for i in range(len(node_ss.shrink_axis_mask)):
                if node_ss.shrink_axis_mask[i] == 1:
                    out_shape.insert(i, 1)
            out_shape = int64_array(out_shape)

        if not np.all(node_ss.new_axis_mask == 0):
            out_shape = list(out_shape)
            for i in reversed(range(len(node_ss.new_axis_mask))):
                if node_ss.new_axis_mask[i] == 1:
                    out_shape.pop(i)
            out_shape = int64_array(out_shape)

        if np.array_equal(input_data_node.shape, out_shape) and \
                all(elem.step == 1 for elem in match['strided_slice'].slices):
            if not np.all(node_ss.shrink_axis_mask == 0):
                ConvertGroupedStridedSlice.add_squeeze_for_shrink(graph, node_ss)
            if not np.all(node_ss.new_axis_mask == 0):
                ConvertGroupedStridedSlice.add_unsqueeze_for_new(graph, node_ss)

            log.info("Useless StridedSlice op '{}' has been detected".format(match['strided_slice'].id))
            # remove inputs to Strided Slice so it has just one input with data so we can use 'remove_op_node' function
            graph.remove_edge(match['strided_slice'].in_node(1).id, match['strided_slice'].id)
            graph.remove_edge(match['strided_slice'].in_node(2).id, match['strided_slice'].id)
            if len(match['strided_slice'].in_nodes()) > 3:
                graph.remove_edge(match['strided_slice'].in_node(3).id, match['strided_slice'].id)

            remove_op_node_with_data_node(graph, match['strided_slice'])
