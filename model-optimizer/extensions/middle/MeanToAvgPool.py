"""
 Copyright (c) 2019 Intel Corporation

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

from mo.graph.graph import create_edge, Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.op import PermuteAttrs, Op
from mo.ops.reshape import Reshape


class MeanToAvgPool(MiddleReplacementPattern):
    enabled = True

    def run_after(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def run_before(self):
        return []

    def pattern(self):
        return dict(
            nodes=[
                ('input', dict(kind='data')),
                ('axis', dict(kind='data')),
                ('mean', dict(kind='op', op='Mean'))
            ],
            edges=[
                ('input', 'mean', {'in': 0}),
                ('axis', 'mean', {'in': 1})
            ]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        if match['axis'].value is None or match['input'].shape is None:
            return
        dims = len(match['input'].shape)
        ones = np.ones(dims, dtype=np.int64)
        axis = np.array(match['axis'].value)
        axis = axis if axis.ndim != 0 else np.array([axis], dtype=np.int64)

        mean = graph.node[match['mean'].node]
        mean['stride'] = np.array(ones)
        # TODO: need to check axis with real layout
        spatial_dims = np.array(axis)
        mean['spatial_dims'] = spatial_dims
        mean['pad'] = np.zeros((dims, 2), np.int64)
        mean['pad_spatial_shape'] = np.array(mean['pad'][spatial_dims])
        window = np.array(ones)
        window[spatial_dims] = match['input'].shape[spatial_dims]
        mean['window'] = window
        mean['TF_op'] = mean['op']
        mean['op'] = 'AvgPool'
        mean['pool_method'] = 'avg'
        mean['rounding_type'] = 'ceil'
        mean['exclude_pad'] = 'true'
        mean['kernel_spatial'] = window[spatial_dims]
        graph.remove_edge(match['axis'].node, match['mean'].node)
        mean['permute_attrs'] = PermuteAttrs().update_attrs(attrs=[('pad', 'input:0'),
                                                                   ('stride', 'input:0'),
                                                                   ('window', 'input:0'),
                                                                   ('spatial_dims', 'input:0')])

        if match['mean'].keep_dims == False:
            output = match['mean'].out_node()
            pool_node = match['mean']

            # Keep dims for AvgPool
            shape = np.array(output.shape)
            for idx in spatial_dims:
                shape = np.insert(shape, idx, 1)

            graph.remove_edge(pool_node.id, output.id)
            # Create new data for pool with all dims
            pool_data = Op.create_data_node(graph, pool_node, {'shape': np.array(shape)})
            # Create and connect reshape node
            reshape_op = Reshape(graph, {'dim': np.array(output.shape)})
            reshape_node = reshape_op.create_node([pool_data], dict(name='Reshape_',
                                                                    permute_attrs=PermuteAttrs().update_attrs(
                                                                        attrs=[('dim', 'output:0')])))
            graph.create_edge(reshape_node, output)
