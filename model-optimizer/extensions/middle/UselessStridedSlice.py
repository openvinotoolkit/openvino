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

import logging as log

import networkx as nx
import numpy as np

from extensions.middle.ConvertGroupedStridedSlice import ConvertGroupedStridedSlice
from extensions.middle.SliceConverter import ConvertSlice
from mo.graph.graph import erase_node
from mo.middle.replacement import MiddleReplacementPattern


class UselessStridedSliceEraser(MiddleReplacementPattern):
    enabled = True

    def run_before(self):
        return [ConvertGroupedStridedSlice]

    def run_after(self):
        return [ConvertSlice]

    def pattern(self):
        return dict(
            nodes=[('strided_slice', dict(kind='op', op='StridedSlice'))],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        output_data_node = match['strided_slice'].out_node(0)
        input_data_node = match['strided_slice'].in_node(0)
        if np.array_equal(input_data_node.shape, output_data_node.shape) and \
                all(elem.step == 1 for elem in match['strided_slice'].slices):
            log.info("Useless StridedSlice op '{}' has been detected".format(match['strided_slice'].id))
            # remove inputs to Strided Slice so it has just one input with data so we can use 'erase_node' function
            graph.remove_edge(match['strided_slice'].in_node(1).id, match['strided_slice'].id)
            graph.remove_edge(match['strided_slice'].in_node(2).id, match['strided_slice'].id)
            graph.remove_edge(match['strided_slice'].in_node(3).id, match['strided_slice'].id)

            erase_node(match['strided_slice'])
            erase_node(output_data_node)
