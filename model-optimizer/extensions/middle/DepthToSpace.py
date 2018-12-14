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
from copy import deepcopy

import networkx as nx

from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.permute import Permute
from mo.ops.reshape import Reshape


class DepthToSpace(MiddleReplacementPattern):
    """
    Replaces DepthToSpace with 6D_Reshape->Permute->4D_Reshape sequence
    """

    enabled = True

    def pattern(self):
        return dict(
            nodes=[
                ('in_data', dict(kind='data')),
                ('op', dict(op='DepthToSpace', data_format='NHWC')),
                ('out_data', dict(kind='data'))
            ],
            edges=[
                ('in_data', 'op'),
                ('op', 'out_data')
            ])

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        node = match['op']

        N, H, W, C = match['in_data'].shape
        block_size = node['block_size']

        graph.remove_edge(match['in_data'].id, match['op'].id)
        graph.remove_edge(match['op'].id, match['out_data'].id)

        dim_6D = [N, block_size, block_size, int(C / (block_size ** 2)), H, W]
        order_6D = [0, 3, 4, 1, 5, 2]
        dim_4D = [N, int(H * block_size), int(W * block_size), int(C / (block_size ** 2))]

        reshape_data_node = Reshape(graph=graph, attrs={'name': match['op'].id + '/Reshape_to_6D', 'dim': dim_6D}).create_node_with_data([match['in_data']])
        permute_data_node = Permute(graph=graph, attrs={'name': match['op'].id + '/Permute', 'order': order_6D}).create_node_with_data([reshape_data_node])
        reshape_node = Reshape(graph=graph, attrs={'infer': None, 'name': match['op'].id + '/Reshape_to_4D', 'dim': dim_4D}).create_node_with_data([permute_data_node], data_nodes=[match['out_data']])

        reshape_data_node.in_node()['nchw_layout'] = True
        reshape_data_node['nchw_layout'] = True
        permute_data_node.in_node()['nchw_layout'] = True
        permute_data_node['nchw_layout'] = True
