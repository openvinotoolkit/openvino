"""
 Copyright (c) 2018 Intel Corporation

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


def tf_tile_infer(node):
    shape = node.in_node(0).shape
    if shape is None:
        log.error("Undefined shape for the input tiles for the Tile operation '{}'.".format(node.node))
        return

    tile_array = node.in_node(1).value
    assert (len(shape) == len(tile_array))
    non_one_tile = np.argwhere(tile_array != 1)
    if len(non_one_tile) != 1:
        node['type'] = None
        log.warning("Tile operation with more than one dimension not equal to 1 is not supported")
    elif len(non_one_tile) == 1:
        node['axis'] = non_one_tile[0][0]
        node['tiles'] = tile_array[node['axis']]
        node.graph.remove_edge(node.in_node(1).id, node.id)

    node.out_node().shape = shape * tile_array
    if node.in_node(0).value is not None:
        node.out_node().value = np.tile(node.in_node(0).value, tile_array)
