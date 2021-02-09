"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.graph.graph import Graph, rename_nodes
from mo.middle.replacement import MiddleReplacementPattern
from mo.ops.pooling import Pooling


class PoolV2ToAttributedPool(MiddleReplacementPattern):
    enabled = True

    def find_and_replace_pattern(self, graph: Graph):
        for pool_v2_node in graph.get_op_nodes(op='PoolingV2'):
            pool_v2_name = pool_v2_node.soft_get('name', pool_v2_node.id)

            pool_v1_node = Pooling(graph, {'window': pool_v2_node.in_port(1).data.get_value(),
                                           'stride': pool_v2_node.in_port(2).data.get_value(),

                                           'pad': pool_v2_node.pad,
                                           'spatial_dims': pool_v2_node.spatial_dims,
                                           'auto_pad': pool_v2_node.auto_pad,
                                           'output_spatial_shape': pool_v2_node.output_spatial_shape,
                                           'pad_spatial_shape': pool_v2_node.pad_spatial_shape,

                                           'pool_method': pool_v2_node.pool_method,
                                           'permute_attrs': pool_v2_node.permute_attrs,}).create_node()

            rename_nodes([(pool_v2_node, pool_v2_name + '/to_be_removed'), (pool_v1_node, pool_v2_name)])

            pool_v2_node.in_port(0).get_connection().set_destination(pool_v1_node.in_port(0))
            pool_v2_node.out_port(0).get_connection().set_source(pool_v1_node.out_port(0))
