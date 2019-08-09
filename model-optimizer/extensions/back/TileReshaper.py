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

from extensions.back.ReshapeMutation import ReshapeMutation
from extensions.back.EltwiseBroadcast import EltwiseBroadcast
from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape


class TileReshaper(BackReplacementPattern):
    enabled = True
    force_shape_inference = True

    def run_after(self):
        return [EltwiseBroadcast]

    def run_before(self):
        return [ReshapeMutation]

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
        tile = match['tile']

        assert len(tile.out_nodes()) == 1, "Tile operation {} should have 1 output data node".format(tile.id)
        out_data = tile.out_node()

        assert out_data.has_valid('shape'), 'Output shape is undefined for {} in back phase'.format(tile.id)
        out_shape = out_data.shape

        if out_shape.size != 3:
            return

        assert len(tile.in_nodes()) == 1, "Tile operation {} should have 1 input data node".format(tile.id)
        inp_data = tile.in_node()

        assert inp_data.has_valid('shape'), 'Input shape is undefined for {} in back phase'.format(tile.id)
        inp_shape = inp_data.shape
        new_inp_shape = np.append(inp_shape, [1])

        reshape = Reshape(graph, {'name': tile.name + '/reshape'}).create_node()
        reshape_dim = Const(graph, {'value': new_inp_shape, 'name': reshape.id + '/Dim'}).create_node()
        tile.in_port(0).get_connection().insert_node(reshape)
        reshape.in_port(1).connect(reshape_dim.out_port(0))

        reshape_back = Reshape(graph, {'name': tile.name + '/reshape_back'}).create_node()
        reshape_back_dim = Const(graph, {'value': out_shape, 'name': reshape.id + '/Dim'}).create_node()
        tile.out_port(0).get_connection().insert_node(reshape_back)
        reshape_back.in_port(1).connect(reshape_back_dim.out_port(0))

        # run shape inference manually for several nodes to override shapes of the model nodes which changed behaviour
        reshape_dim.infer(reshape_dim)
        reshape.infer(reshape)
        tile.infer(tile)
