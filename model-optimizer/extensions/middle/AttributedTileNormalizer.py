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
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.const import Const
from mo.ops.tile import Tile


class AttributedTileNormalizer(MiddleReplacementPattern):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('tile', dict(op='AttributedTile', axis=lambda x: x is not None, tiles=lambda x: x is not None))],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['tile']
        name = node.soft_get('name', node.id)

        axis = node.axis
        tiles = node.tiles

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None
        tiles_input_value = int64_array(np.ones(input_shape.size))
        tiles_input_value[axis] = tiles

        const = Const(graph, {'value': tiles_input_value, 'name': name + '/tiles'}).create_node()
        tile = Tile(graph, {'name': name}).create_node()

        node.out_port(0).get_connection().set_source(tile.out_port(0))
        node.in_port(0).get_connection().set_destination(tile.in_port(0))
        const.out_port(0).connect(tile.in_port(1))
