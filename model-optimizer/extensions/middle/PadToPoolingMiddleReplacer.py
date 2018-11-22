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

import numpy as np
import networkx as nx

from mo.ops.pooling import Pooling
from mo.graph.graph import unique_id
from mo.middle.replacement import MiddleReplacementPattern
from mo.front.common.layout import get_features_dim

class PadToPoolingMiddleReplacer(MiddleReplacementPattern):
    op = "Pad"
    enabled = False

    def pattern(self):
        return dict(
            nodes=[
                ('pad', dict(kind='op', op='Pad'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: nx.MultiDiGraph, match: dict):
        node = match['pad']
        input = node.in_node()
        output = node.out_node()
        if len(output.out_nodes()) > 0:
            ndim = len(input.shape)
            pad = node.pads
            graph.remove_edge(input.id, node.id)
            graph.remove_edge(node.id, output.id)
            pool_node = unique_id(graph, node.name + '/Pool_')
            Pooling(graph, dict(name=pool_node, window=np.ones(ndim, dtype=np.int64),
                                output_spatial_shape=None,
                                batch_dims=np.array([0], dtype=np.int64),
                                channel_dims=np.array([get_features_dim(graph.graph['layout'], ndim)], dtype=np.int64),
                                stride=np.array(np.ones(ndim, dtype=np.int64)),
                                pad=pad, exclude_pad='false', pool_method='max')).create_node_with_data(inputs=[input], data_nodes=[output])
