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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.reshape import Reshape
from mo.ops.tile import Tile
from mo.ops.unsqueeze import Unsqueeze


class MXRepeatReplacer(MiddleReplacementPattern):
    """
        The transformation converts MXRepeat operation to Unsqueeze -> Tile -> Reshape.
    """

    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('repeat', dict(kind='op', op='MXRepeat'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        mxrepeat = match['repeat']

        out_shape = mxrepeat.out_node(0).shape
        unsqueeze_node = create_op_node_with_second_input(graph, Unsqueeze, int64_array([mxrepeat.axis+1]),
                                                        dict(name=mxrepeat.id + "/ExpandDims", expand_axis=int64_array([mxrepeat.axis])))

        tile_array = np.ones([len(mxrepeat.in_node().shape) + 1], dtype=np.int64)
        tile_array[mxrepeat.axis + 1] = mxrepeat.repeats
        tile_node = create_op_node_with_second_input(graph, Tile, int64_array(tile_array),
                                                     dict(name=mxrepeat.id + "/Tile", axis=mxrepeat.axis + 1))

        reshape_node = create_op_node_with_second_input(graph, Reshape, int64_array(out_shape),
                                                     dict(name=mxrepeat.id + "/Reshape"))

        mxrepeat.in_port(0).get_connection().set_destination(unsqueeze_node.in_port(0))
        tile_node.in_port(0).connect(unsqueeze_node.out_port(0))
        reshape_node.in_port(0).connect(tile_node.out_port(0))
        mxrepeat.out_port(0).get_connection().set_source(reshape_node.out_port(0))
