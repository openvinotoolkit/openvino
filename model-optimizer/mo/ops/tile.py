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

import logging as log
import numpy as np

from mo.graph.graph import Node, Graph
from mo.ops.op import Op, PermuteAttrs


class Tile(Op):
    op = 'Tile'
    enabled = True

    def __init__(self, graph: Graph, attrs: dict):
        super().__init__(graph, {
            'kind': 'op',
            'type': __class__.op,
            'op': __class__.op,
            'in_ports_count': 1,
            'out_ports_count': 1,
            'infer': Tile.infer
        }, attrs)

    def supported_attrs(self):
        return ['axis', 'tiles']

    @staticmethod
    def infer(node: Node):
        shape = node.in_node().shape
        if shape is None:
            log.error("Undefined shape for the input tiles for the Tile operation '{}'.".format(node.node))
            return
        shape = np.copy(shape)

        if len(node.in_nodes()) == 2:
            tile_array = node.in_node(1).value
            if tile_array is None:
                log.error('A tile values are None for a node "{}".'.format(node.name))
                return
            if len(shape) != len(tile_array):
                log.error('Shape mismatch for a node "{}": {} vs {}.'.format(node.name, shape.shape, tile_array.shape))
                return
            non_one_tile = np.argwhere(tile_array != 1)
            if len(non_one_tile) == 0:
                log.info(
                    'Redundant "Tile" operation "{}" with tile values for all dimensions equal to 1.'.format(node.name))
                node['axis'] = 0
                node['tiles'] = 1
            elif len(non_one_tile) == 1:
                node['axis'] = non_one_tile[0][0]
                node['tiles'] = tile_array[node['axis']]
            else:
                node['type'] = None
                node['tile_array'] = tile_array
                log.warning("Tile operation with more than one dimension not equal to 1 is not supported.")
                # do not return here to allow infer shape and values for the constant propagation case
            node.graph.remove_edge(node.in_node(1).id, node.id)
        elif len(node.in_nodes()) == 1:  # case when tiled dimension and count are specified in node attributes
            if not node.has_valid('axis') or not node.has_valid('tiles'):
                log.error('Mandatory attributes "axis" or "tiles" are not specified for a Tile node "{}"'.
                          format(node.name))
                return
            tile_array = np.ones([len(shape)], dtype=np.int64)
            tile_array[node.axis] = node.tiles
        else:
            log.error('Unsupported number of input parameters to Tile node "{}"'.format(node.name))
            return

        PermuteAttrs.create_permute_attrs(node, attrs=[('axis', 'input:0')])
        node.out_node().shape = shape * tile_array
        if node.in_node(0).value is not None:
            node.out_node().value = np.tile(node.in_node(0).value, tile_array)
