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

from mo.front.caffe.extractors.utils import get_canonical_axis_index
from mo.front.common.layout import get_batch_dim, get_features_dim
from mo.graph.graph import Graph
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.pooling import Pooling
from mo.ops.power import Power
from mo.ops.reshape import Reshape


class ReduceReplacer(MiddleReplacementPattern):
    op = "Reduce"
    enabled = True

    supported_reduce_types = ['mean', 'max', 'sum']

    pool_method_map = {
        'max': 'max',
        'mean': 'avg',
        'sum': 'avg'
    }

    def run_after(self):
        from extensions.middle.pass_separator import MiddleStart
        return [MiddleStart]

    def run_before(self):
        from extensions.middle.pass_separator import MiddleFinish
        return [MiddleFinish]

    def pattern(self):
        return dict(
            nodes=[
                ('reduce', dict(kind='op', op='Reduce'))
            ],
            edges=[]
        )

    def replace_pattern(self, graph: Graph, match: dict):
        node = match['reduce']
        if not node.has_valid('reduce_type') or node.reduce_type.lower() not in self.supported_reduce_types:
            log.error("Reduce type {} is not supported for node {}".format(node.soft_get('reduce_type'), node.id))
            return

        reduce_type = node.reduce_type.lower()
        if reduce_type not in self.pool_method_map:
            log.error("Reduce type {} is not included in pool_method_map. Please update pool_method_map with new key "
                      "{}".format(reduce_type, reduce_type))
            return

        input_data = node.in_node()
        output_data = node.out_node()

        input_shape = node.in_node().shape
        output_shape = node.out_node().shape

        # normalize node.axis to exclude negative indices
        node.axis = [get_canonical_axis_index(input_shape, a) for a in node.axis]

        axis = node.axis

        # Check that values in axis list are consecutive
        for idx in range(1, len(axis)):
            if axis[idx] != (axis[idx - 1] + 1):
                log.error("Reduce with not consecutive axes {} is not supported ".format(axis))
                return

        layout = graph.graph['layout']

        # So now we are sure that we can convert Reduce to appropriate operation

        # 1. Calculate shape that will be used in reduction
        reduction_dim = np.prod([input_shape[idx] for idx in axis])
        begin_dims = np.array([input_shape[idx] for idx in range(axis[0])])
        end_dim = np.prod([input_shape[idx] for idx in range(axis[-1] + 1, len(input_shape))])

        # 2. Create reshape with appropriate shape
        if layout == 'NCHW':
            if len(begin_dims) > 2:
                begin_dims = np.array([np.prod(begin_dims[0:-1]), begin_dims[-1]], dtype=np.int64)
            else:
                # Expand begin_dims to 2
                begin_dims = np.array(np.append(begin_dims, [1] * (2 - len(begin_dims))), dtype=np.int64)
            reshape_shape = np.array([*begin_dims, reduction_dim, end_dim], dtype=np.int64)
            pool_window = np.array([1, 1, reduction_dim, 1], dtype=np.int64)
        elif layout == 'NHWC':
            begin_dims = np.prod(begin_dims)
            reshape_shape = np.array([begin_dims, reduction_dim, 1, end_dim], dtype=np.int64)
            pool_window = np.array([1, reduction_dim, 1, 1], dtype=np.int64)
        else:
            log.error('{} layout currently is not supported'.format(layout))
            return

        # 3. Reduce => Reshape->Pooling->Reshape
        reshape_op = Reshape(graph, {'name': node.id + '/Reshape', 'dim': reshape_shape})
        final_reshape_op = Reshape(graph, {'name': node.id + '/FinalReshape', 'dim': output_shape})
        pooling_op = Pooling(graph,
                             dict(name=node.id + '/Pool',
                                  window=pool_window,
                                  output_spatial_shape=None,
                                  batch_dims=np.array([get_batch_dim(layout, 4)], dtype=np.int64),
                                  channel_dims=np.array([get_features_dim(layout, 4)], dtype=np.int64),
                                  exclude_pad='false', pool_method=self.pool_method_map[reduce_type]))

        graph.remove_edge(input_data.id, node.id)
        graph.remove_edge(node.id, output_data.id)

        final_reshape_op.create_node_with_data(
            inputs=[pooling_op.create_node_with_data(
                inputs=[reshape_op.create_node_with_data(
                    inputs=[input_data]
                )]
            )],
            data_nodes=output_data)

        # 4. If it is reduction with summation, we need to multiply by size of the reduction slice with Mul op
        if reduce_type == 'sum':
            output_data.in_node().insert_node_with_data_after(
                output_data,
                Power,
                {'name': node.name + '/Mul', 'scale': float(reduction_dim)}
            )
