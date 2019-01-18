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

import networkx as nx
import numpy as np

from extensions.back.ConvolutionReshaper import ConvolutionReshaper
from extensions.back.TileReshaper import TileReshaper
from mo.back.replacement import BackReplacementPattern
from mo.front.common.layout import get_width_dim, get_height_dim, get_features_dim, indices_mapping
from mo.ops.op import PermuteAttrs
from mo.ops.permute import Permute


class PermuteForReshape(BackReplacementPattern):
    """
       Fixes problem with Reshapes that changes shape of tensor from >= 4D tensor
       (where permutation works) to 3D (where permutation doesn't work since we are not sure in new layout).
       that leads to wrong shapes after permutations (since one part of shapes is permuted while other isn't).
    """
    enabled = True

    def run_before(self):
        return [ConvolutionReshaper,
                TileReshaper,
                ]

    @staticmethod
    def pattern():
        return dict(
            nodes=[('reshape', {'kind': 'op', 'type': 'Reshape'})],
            edges=[],
        )

    @staticmethod
    def replace_pattern(graph: nx.MultiDiGraph, match: dict):
        reshape = match['reshape']
        assert len(reshape.in_nodes()) > 0
        if graph.graph['layout'] == 'NCHW' or reshape.has_and_set('nchw_layout') or\
                reshape.soft_get('correct_data_layout') is True:
            return

        input_node = reshape.in_node()
        output_node = reshape.out_node()
        input_shape = input_node.shape
        output_shape = output_node.shape

        if len(input_shape) >= 4 and len(output_shape) == 3:
            # Check that we will permute some shapes in this Reshape by our permutation pass
            layout = 'NCHW'
            c_idx = get_features_dim(layout, len(input_shape))
            hw_idx = [get_width_dim(layout, len(input_shape)), get_height_dim(layout, len(input_shape))]
            if input_shape[c_idx] != 1 and np.any(input_shape[hw_idx] != [1, 1]):
                # then nhwc -> nchw permutation can change shapes significantly
                # We need to wrap up node with NCHW -> NHWC permutes and don't touch it later
                permutation = PermuteAttrs.get_nchw_to_nhwc_permutation(len(input_shape))
                permutation_back = PermuteAttrs.get_nchw_to_nhwc_permutation(len(input_shape))

                # 1. Insert input Permute
                #    This Permute will permute input from original input layout to operation layout
                edge_attrs = graph.get_edge_data(input_node.id, reshape.id)[0]
                graph.remove_edge(input_node.id, reshape.id)

                permute_op = Permute(graph, {'order': permutation.perm, 'name': reshape.name + '/Permute_'})
                permute_data_node = permute_op.create_node_with_data([input_node])

                graph.add_edge(permute_data_node.id, reshape.id, **edge_attrs)