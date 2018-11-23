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

import networkx as nx
import numpy as np

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.crop import Crop
from mo.ops.op import Op


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

    def pattern(self):
        return dict(
            nodes=[
                ('slice',dict(kind='op', op='Slice'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        node = match['slice']
        # Caffe case
        if not node.has_valid('start') or not node.has_valid('end'):
            return

        begin = node.start
        end = node.end

        input = node.in_node(0)
        output_data = node.out_node()

        # Check whether operation use only one axis or not
        dims = 0
        axes = np.zeros(begin.size)
        for i in range(begin.size):
            if begin[i] != 0 or end[i] != input.shape[i]:
                dims += 1
                axes[i] = 1
        axes = np.array(axes, dtype=bool)
        if dims == 0:
            return
        elif dims == 1:
            # If Slice use only one axis, than
            # convert Slice to StridedSlice
            node['op'] = 'StridedSlice'
            node['type'] = 'StridedSlice'

            convert_negative_indices(begin, input.shape)
            convert_negative_indices(end, input.shape)
        else:
            # If Slice use more than one axis use Crop layer
            crop = Crop(graph, dict(axis=np.arange(begin.size)[axes],
                                    offset=begin[axes]))
            # creating node with data
            crop.create_node_with_data(inputs=[input], data_nodes=[output_data])

            # Remove unnecessary edges from and to to Slice vertex
            graph.remove_edge(input.id, node.id)
            graph.remove_edge(node.id, output_data.id)
