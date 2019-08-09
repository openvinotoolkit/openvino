"""
 Copyright (c) 2019 Intel Corporation

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
from mo.ops.tile import Tile


class TileMultipleAxisReplacer(MiddleReplacementPattern):
    """
        This class convert Tile operation with miltiple != 1 tile dimensions by sequence of Tiles.
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
        tile = match['tile']

        if tile.has_valid('tile_array'):
            tile_array = tile.tile_array
            assert len(tile_array) == len(tile.in_port(0).data.get_shape())

            non_one_tile = np.argwhere(tile_array != 1).flatten()

            # We need to add new tiles only in case when we tile more than one dimension
            if len(non_one_tile) > 1:
                last_tile = None
                for i in non_one_tile:
                    axis = i
                    tiles = tile_array[i]
                    new_tile = Tile(graph, {'name': tile.name + '/Tile_{}/'.format(i), 'axis': axis, 'tiles': tiles,
                                            'need_shape_inference': True}).create_node()
                    if not last_tile:
                        last_tile = new_tile
                        tile.in_port(0).get_connection().set_destination(new_tile.in_port(0))
                    else:
                        last_tile.out_port(0).connect(new_tile.in_port(0))
                        last_tile = new_tile

                # Reconnect output to new tile node and delete old tile
                tile.out_port(0).get_connection().set_source(last_tile.out_port(0))
