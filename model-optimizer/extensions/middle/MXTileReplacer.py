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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.tf.graph_utils import create_op_node_with_second_input
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.unsqueeze import Unsqueeze


class MXTileReplacer(MiddleReplacementPattern):
    """
        Aligns Tile operation from MxNet framework with OpenVINO Tile

        MxNet has no restrictions for `tile_array` input of `Tile` operation.
        If len(tile_array) > rank(data), this transformation will insert Unsqueeze before Tile operation,
        because in this case output_shape > input_shape

        DOC link: https://beta.mxnet.io/api/ndarray/_autogen/mxnet.ndarray.tile.html#mxnet.ndarray.tile
    """

    enabled = True

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
        in_shape = node.in_port(0).data.get_shape()
        out_shape = node.out_port(0).data.get_shape()

        tile_array_diff = len(out_shape) - len(in_shape)
        if tile_array_diff == 0:
            return
        assert tile_array_diff > 0,\
            'Unexpected difference between rank(input) and rank(output) for node {}'.format(name)
        unsqueeze_dims = int64_array(range(tile_array_diff))
        unsqueeze = create_op_node_with_second_input(graph, Unsqueeze, unsqueeze_dims,
                                                     dict(name=name + '/Unsqueeze', override_output_shape=True))
        node.in_port(0).get_connection().insert_node(unsqueeze)
