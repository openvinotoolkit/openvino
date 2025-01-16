# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.replacement import MiddleReplacementPattern
from openvino.tools.mo.ops.const import Const
from openvino.tools.mo.ops.tile import Tile


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
