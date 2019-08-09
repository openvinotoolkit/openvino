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

import networkx as nx
import numpy as np

from mo.back.replacement import BackReplacementPattern
from mo.graph.graph import Node, Graph
from mo.ops.tile import Tile
from mo.ops.broadcast import Broadcast
from mo.ops.const import Const


class EltwiseBroadcast(BackReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].generate_experimental_IR_V10]

    @staticmethod
    def pattern():
        return dict(
            nodes=[
                ('op', dict(kind='op', is_eltwise=True))],
            edges=[]
        )

    @staticmethod
    def replace_pattern(graph: Graph, match: dict):
        node = match['op']
        shapes = [in_node.shape for _, in_node in node.in_nodes().items()]
        out_shape = node.out_node().shape
        broadcast_name = node.name + '/Broadcast/'

        for i, shape in enumerate(shapes):
            if not np.array_equal(shape, out_shape):
                # Add Broadcast op for this input
                # Need to create additional Const op for shape
                new_shape = Const(graph, {'name': broadcast_name + 'Shape', 'value': out_shape.copy()}).create_node()
                broadcast_axis = Const(graph, {
                    'name': broadcast_name + 'Axis',
                    'value': np.array(range(len(out_shape)), dtype=np.int64)}
                ).create_node()
                broadcast = Broadcast(graph, {'name': broadcast_name}).create_node()
                node.in_port(i).get_connection().set_destination(broadcast.in_port(0))
                broadcast.in_port(1).connect(new_shape.out_port(0))
                broadcast.in_port(2).connect(broadcast_axis.out_port(0))
                broadcast.out_port(0).connect(node.in_port(i))
