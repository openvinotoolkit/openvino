# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node, Graph
from mo.graph.perm_inputs import PermuteInputs
from mo.ops.op import Op, PermuteAttrs


class Tile(Op):
    op = 'Tile'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': self.op,
            'version': 'opset1',

            'infer': self.infer,

            'in_ports_count': 2,
            'out_ports_count': 1,
        }, attrs)

    @staticmethod
    def infer(node: Node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) == 2 and 0 in connected_in_ports and 1 in connected_in_ports, \
            "Tile should have 2 connected input port, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_in_ports)

        shape = node.in_port(0).data.get_shape()
        assert shape is not None, "Undefined input shape for Tile node '{}'.".format(name)
        tile_array = node.in_port(1).data.get_value()
        assert tile_array is not None, "Undefined `repeats` (1st port input value) of Tile node '{}'".format(name)

        # align ranks of the tile_array tensor and input shape node
        if shape.size < tile_array.size:
            shape = np.insert(shape, 0, [1] * (tile_array.size - shape.size))
        elif shape.size > tile_array.size:
            tile_array = np.insert(tile_array, 0, [1] * (shape.size - tile_array.size))

        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.tile(node.in_port(0).data.get_value().reshape(shape), tile_array))
        else:
            node.out_port(0).data.set_shape(shape * tile_array)

        PermuteInputs().set_input_permutation(node.in_node(1), node, 'input:0', 'shape')


class AttributedTile(Op):
    op = 'AttributedTile'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'op': self.op,
            'type': 'Tile',
            'version': 'opset1',

            'infer': self.infer,

            'in_ports_count': 1,
            'out_ports_count': 1,
        }, attrs)

        assert 'axis' in self.attrs
        assert 'tiles' in self.attrs

    def supported_attrs(self):
        return ['axis', 'tiles']

    @staticmethod
    def infer(node):
        name = node.soft_get('name', node.id)

        connected_in_ports = {idx: port for idx, port in node.in_ports().items() if not port.disconnected()}
        assert len(connected_in_ports) == 1 and 0 in connected_in_ports, \
            "AttributedTile should have 1 connected input port, but it doesn't for node: `{}`. Ports: {}" \
            "".format(name, connected_in_ports)

        shape = node.in_port(0).data.get_shape()
        assert shape is not None, "Undefined input shape for AttributedTile node '{}'.".format(name)
        axis = node.soft_get('axis', None)
        assert axis is not None
        tiles = node.soft_get('tiles', None)
        assert tiles is not None, "Undefined `tiles` attribute of Tile node '{}'".format(name)

        tile_array = int64_array(np.ones(shape.size))
        tile_array[node.axis] = node.tiles

        node.out_port(0).data.set_shape(shape * tile_array)
        if node.in_port(0).data.get_value() is not None:
            node.out_port(0).data.set_value(np.tile(node.in_port(0).data.get_value(), tile_array))

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
