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
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.tile import Tile


class MXTileReplacer(MiddleReplacementPattern):
    """
        This class Reshape Tile operation if len input shape < output shape.
    """

    enabled = True
    force_clean_up = True

    def pattern(self):
        return dict(
            nodes=[
                ('tile', dict(kind='op', op='Tile'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        mxtile = match['tile']

        in_shape = mxtile.in_port(0).data.get_shape()
        out_shape = mxtile.out_node(0).shape

        tile_array_diff = (len(out_shape) - len(in_shape))
        if tile_array_diff > 0:
            reshape_shape = np.copy(in_shape)
            for i in range(tile_array_diff):
                reshape_shape = np.insert(in_shape, 0, 1, axis=0)
            reshape_node = create_op_node_with_second_input(graph, Reshape, int64_array(reshape_shape), dict(name=mxtile.id + "/Reshape"))
            mxtile.in_port(0).get_source().get_connection().set_destination(reshape_node.in_port(0))
            reshape_node.out_port(0).get_connection().set_destination(mxtile.in_port(0))
