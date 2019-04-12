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

import numpy as np

from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.crop import Crop
from mo.ops.strided_slice import StridedSlice


def convert_negative_indices(indices: np.array, shape: np.array):
    for ind, value in enumerate(indices):
        if value < 0:
            indices[ind] += shape[ind]


class ConvertSlice(MiddleReplacementPattern):
    """
    This class convert Slice operation to Crop or Split depends on parameters
    """

    enabled = True
    op = "Slice"

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def pattern(self):
        return dict(
            nodes=[
                ('slice', dict(kind='op', op='Slice'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['slice']
        # Caffe case
        if not node.has_valid('start') or not node.has_valid('end'):
            return

        begin = node.start
        end = node.end
        axis = node.axis if node.has_valid('axis') else range(begin.size)
        

        input = node.in_node(0)
        output_data = node.out_node()

        # Check whether operation use only one axis or not
        axes_begin = np.zeros(len(input.shape), dtype=np.int32)
        axes_end = np.zeros(len(input.shape), dtype=np.int32)
        begin_ext = np.zeros(len(input.shape), dtype=np.int32)
        end_ext = np.zeros(len(input.shape), dtype=np.int32)
        dims = 0
        axes = np.zeros(begin.size)
        for i in range(len(axis)):
            if begin[i] != 0 or end[i] < input.shape[i]:
                dims += 1
                axes[i] = 1
                if begin[i] != 0:
                    axes_begin[axis[i]] = 1
                    begin_ext[axis[i]] = begin[i]
                if end[i] < input.shape[i]:
                    axes_end[axis[i]] = 1
                    end_ext[axis[i]] = end[i]
        axes = np.array(axes, dtype=bool)

        if dims == 1 or dims == 0:
            # If Slice use only one axis or no axis, than
            # convert Slice to StridedSlice
            ss = StridedSlice(graph, dict(new_axis_mask=np.zeros(len(output_data.shape), dtype=np.int32),
                                          shrink_axis_mask=np.zeros(len(output_data.shape), dtype=np.int32),
                                          ellipsis_mask=np.zeros(len(output_data.shape), dtype=np.int32),
                                          begin_mask=axes_begin,
                                          end_mask=axes_end))

            convert_negative_indices(begin_ext, input.shape)
            convert_negative_indices(end_ext, input.shape)

            begin_node = Const(graph, {'name': 'begin', 'value': begin_ext, 'force_precision': 'I32'}).create_node_with_data()
            end_node = Const(graph, {'name': 'end', 'value': end_ext, 'force_precision': 'I32'}).create_node_with_data()

            ss.create_node_with_data(inputs=[input, begin_node, end_node], data_nodes=[output_data])
            # Remove unnecessary edges from and to to Slice vertex
            graph.remove_edge(input.id, node.id)
            graph.remove_edge(node.id, output_data.id)
        else:
            # If Slice use more than one axis use Crop layer
            crop = Crop(graph, dict(axis=np.arange(begin.size)[axes],
                                    offset=begin[axes]))
            # creating node with data
            crop.create_node_with_data(inputs=[input], data_nodes=[output_data])

            # Remove unnecessary edges from and to to Slice vertex
            graph.remove_edge(input.id, node.id)
            graph.remove_edge(node.id, output_data.id)
