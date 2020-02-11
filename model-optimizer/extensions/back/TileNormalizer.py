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

from extensions.back.ReshapeMutation import ReshapeMutation
from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.ops.concat import Concat
from mo.ops.const import Const
from mo.ops.squeeze import Squeeze
from mo.ops.tile import Tile, AttributedTile
from mo.ops.unsqueeze import Unsqueeze
from mo.utils.shape import new_shape_node_from_shape_nodes


class TileInputAlignment(BackReplacementPattern):
    """
    Aligns rank of data input and length of repeats input of Tile operation
    """
    enabled = True
    force_clean_up = True
    force_shape_inference = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def pattern(self):
        return dict(
            nodes=[
                ('tile', dict(kind='op', op='Tile'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['tile']
        name = node.soft_get('name', node.id)

        input_shape = node.in_port(0).data.get_shape()
        assert input_shape is not None
        tiles = node.in_port(1).data.get_value()
        assert tiles is not None, "Undefined `repeats` (1st port input value) of Tile node '{}'".format(name)

        if input_shape.size == tiles.size:
            return

        if input_shape.size < tiles.size:
            unsqueeze = create_op_node_with_second_input(graph, Unsqueeze,
                                                         int64_array(list(range(tiles.size - input_shape.size))),
                                                         {'name': name + '/input_alignment',
                                                          'override_output_shape': True})
            node.in_port(0).get_source().connect(unsqueeze.in_port(0))
            node.in_port(0).get_connection().set_source(unsqueeze.out_port(0))
        else:
            const = Const(graph, {'name': name + '/tile_alignment_const',
                                  'value': np.ones([input_shape.size - tiles.size], dtype=np.int64)}).create_node()
            concat = Concat(graph, {'axis': 0, 'override_output_shape': True}).create_node()
            concat.add_input_port(0)
            concat.add_input_port(1)

            node.in_port(1).get_source().connect(concat.in_port(1))
            node.in_port(1).disconnect()
            concat.in_port(0).connect(const.out_port(0))

            node.in_port(1).connect(concat.out_port(0))


class Tile3DReshaper(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_before(self):
        return [ReshapeMutation, TileVersionDowngrader]

    def run_after(self):
        return [TileInputAlignment]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('tile', dict(type='Tile'))
            ],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        """
        Workarounds not supported type of Tile in Inference Engine (Tiles are supported for 2-D or 4-D tensors):
        Searches for Tiles with 3D shapes and covers it with Reshapes.

        Example: Tile (axis=1, tiles=16):
            in_shape: [1,1,101]
            out_shape: [1,16,101]

        Old behaviour:
            Tile -> [1,16,101]
        New behaviour:
            Reshape [1,1,101,1] -> Tile -> [1,16,101,1] -> Reshape [1,16,101]
        """
        node = match['tile']
        name = node.soft_get('name', node.id)

        out_shape = node.out_port(0).data.get_shape()
        assert out_shape is not None, 'Output shape is undefined for {} in back phase'.format(name)
        if out_shape.size != 3:
            return

        inp_shape = node.in_port(0).data.get_shape()
        assert inp_shape is not None, 'Input shape is undefined for {} in back phase'.format(name)

        unsqueeze_dim = Const(graph, {'name': name + '/3D_Tile_Unsqueeze_dim', 'value': int64_array([3])}).create_node()
        unsqueeze = Unsqueeze(graph, {'name': name + '/3D_Tile_Unsqueeze', 'override_output_shape': True}).create_node()
        unsqueeze_dim.out_port(0).connect(unsqueeze.in_port(1))

        const = Const(graph, {'name': name + '/additional_axis', 'value': int64_array([1])}).create_node()
        new_tiles = new_shape_node_from_shape_nodes([node.in_port(1).get_source().node, const])

        node.in_port(1).get_connection().set_source(new_tiles.out_port(0))

        squeeze_dim = Const(graph, {'name': name + '/3D_Tile_Squeeze_dim', 'value': int64_array([3])}).create_node()
        squeeze = Squeeze(graph, {'name': name + '/3D_Tile_Squeeze', 'override_output_shape': True}).create_node()
        squeeze_dim.out_port(0).connect(squeeze.in_port(1))

        source = node.in_port(0).get_source()
        node.in_port(0).get_connection().set_source(unsqueeze.out_port(0))
        unsqueeze.in_port(0).connect(source)

        node.out_port(0).get_connection().set_source(squeeze.out_port(0))
        node.out_port(0).connect(squeeze.in_port(0))

        node['override_output_shape'] = True
        new_tiles['override_output_shape'] = True
        node['need_shape_inference'] = True


class TileMultipleAxisReplacer(BackReplacementPattern):
    """
        This class convert Tile operation with miltiple != 1 tile dimensions by sequence of Tiles.
    """
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_after(self):
        return [TileInputAlignment]

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
        name = tile.soft_get('name', tile.id)

        tile_array = tile.in_port(1).data.get_value()
        assert tile_array is not None, "Undefined `repeats` (1st port input value) of Tile node '{}'".format(name)
        assert len(tile_array) == len(tile.in_port(0).data.get_shape())

        non_one_tile = np.argwhere(tile_array != 1).flatten()
        if len(non_one_tile) == 1:
            # We need to add new tiles only in case when we tile more than one dimension
            return

        if len(non_one_tile) == 0:
            # Deleting such Tile that does nothing
            tile.out_port(0).get_connection().set_source(tile.in_port(0).get_connection().get_source())
            return

        last_tile = None
        for i in non_one_tile:
            tiles = int64_array(np.ones(tile_array.size))
            tiles[i] = tile_array[i]
            new_tile = create_op_node_with_second_input(graph, Tile, tiles, {'name': name + '/Tile_{}/'.format(i)})

            if not last_tile:
                last_tile = new_tile
                tile.in_port(0).get_connection().set_destination(new_tile.in_port(0))
            else:
                last_tile.out_port(0).connect(new_tile.in_port(0))
                last_tile = new_tile

        # Reconnect output to new tile node and delete old tile
        tile.out_port(0).get_connection().set_source(last_tile.out_port(0))


class TileVersionDowngrader(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    graph_condition = [lambda graph: not graph.graph['cmd_params'].generate_experimental_IR_V10]

    def run_after(self):
        return [TileMultipleAxisReplacer]

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
        name = tile.soft_get('name', tile.id)

        tile_array = tile.in_port(1).data.get_value()
        assert tile_array is not None, "Undefined `repeats` (1st port input value) of Tile node '{}'".format(name)

        non_one_tile = np.argwhere(tile_array != 1).flatten()
        assert len(non_one_tile) == 1

        axis = non_one_tile[0]
        tiles = tile_array[axis]

        new_tile = AttributedTile(graph, {'name': name, 'axis': axis, 'tiles': tiles}).create_node()

        tile.out_port(0).get_connection().set_source(new_tile.out_port(0))
        tile.in_port(0).get_connection().get_source().connect(new_tile.in_port(0))
