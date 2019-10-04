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
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph
from mo.ops.const import Const
from mo.ops.reshape import Reshape
from mo.ops.squeeze import Squeeze
from mo.ops.unsqueeze import Unsqueeze


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
        node = match['tile']
        name = node.soft_get('name', node.id)

        out_shape = node.out_port(0).data.get_shape()
        assert out_shape is not None, 'Output shape is undefined for {} in back phase'.format(name)
        if out_shape.size != 3:
            return

        inp_shape = node.in_port(0).data.get_shape()
        assert inp_shape is not None, 'Input shape is undefined for {} in back phase'.format(name)

        unsqueeze_dim = Const(graph, {'name': name + '/3D_Tile_Unsqueeze_dim', 'value': int64_array([3])}).create_node()
        unsqueeze = Unsqueeze(graph, {'name': name + '/3D_Tile_Unsqueeze'}).create_node()
        unsqueeze_dim.out_port(0).connect(unsqueeze.in_port(1))

        squeeze_dim = Const(graph, {'name': name + '/3D_Tile_Squeeze_dim', 'value': int64_array([3])}).create_node()
        squeeze = Squeeze(graph, {'name': name + '/3D_Tile_Squeeze'}).create_node()
        squeeze_dim.out_port(0).connect(squeeze.in_port(1))

        source = node.in_port(0).get_source()
        node.in_port(0).get_connection().set_source(unsqueeze.out_port(0))
        unsqueeze.in_port(0).connect(source)

        node.out_port(0).get_connection().set_source(squeeze.out_port(0))
        node.out_port(0).connect(squeeze.in_port(0))

        # run shape inference manually for several nodes to override shapes of the model nodes which changed behaviour
        unsqueeze_dim.infer(unsqueeze_dim)
        unsqueeze.infer(unsqueeze)
        node.infer(node)
        squeeze_dim.infer(squeeze_dim)
        squeeze.infer(squeeze)

