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

import networkx as nx
import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import unique_id, Node
from mo.ops.tile import Tile


class EltwiseBroadcast(BackReplacementPattern):
    enabled = True

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', type='Eltwise'))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        node = match['op']
        shapes = [in_node.shape for _, in_node in node.in_nodes().items()]
        out_shape = node.out_node().shape
        tname = node.name + '/Broadcast/'
        tile = Tile(graph, dict(name=tname))

        # Working with scalar values
        for i, shape in enumerate(shapes):
            if len(shape) == 0:
                shapes[i] = np.ones(len(out_shape), dtype=np.int64)
                node.in_node(i).shape = shapes[i].copy()
                if node.in_node(i).value is not None:
                    node.in_node(i).value = np.reshape(node.in_node(i).value, newshape=shapes[i])

        if not all([len(shape) == len(out_shape) for shape in shapes]):
            log.warning("Cannot apply broadcast for Eltwise layer {} "
                        "because not all input shapes {} have the same number of elements "
                        "as output shape {}.".format(node.soft_get('name'),
                                                     shapes,
                                                     out_shape
                                                     )
                        )
            return

        input_idx = 0
        for port, old_input in node.in_nodes().items():
            # old_input = node.in_node(input_idx)
            input = old_input
            for i in range(len(out_shape)):
                if shapes[input_idx][i] == 1 and out_shape[i] > 1:
                    new_op = tile.create_node([input], dict(axis=i, tiles=out_shape[i]))
                    # add a data node following a new operation node
                    data_id = unique_id(graph, node.name)
                    graph.add_node(data_id, kind='data', shape=None, value=None)
                    new_data = Node(graph, data_id)
                    graph.add_edge(new_op.id, new_data.id, **{'out': 0})
                    new_op.infer(new_op)
                    input = new_data
            if input != old_input:
                # create a new edge from new data node after Tile application to the eltwise
                # and copy all edge attributes from the old edge
                # [0] is not what we really want
                graph.add_edge(input.id, node.id, **graph[old_input.id][node.id][0])
                graph.remove_edge(old_input.id, node.id)
            input_idx += 1
