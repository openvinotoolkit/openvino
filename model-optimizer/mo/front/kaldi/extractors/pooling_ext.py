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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.ops.op import Op


class PoolingFrontExtractor(FrontExtractorOp):
    op = 'pooling'
    enabled = True

    @staticmethod
    def extract(node):
        mapping_rule = {
            'window': int64_array([1, 1, node.pb.kernel, 1]),
            'stride': int64_array([1, 1, node.pb.stride, node.pb.stride]),
            'pool_stride': node.pb.pool_stride,
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'infer': PoolingFrontExtractor.infer
        }
        mapping_rule.update(layout_attrs())
        Op.get_op_class_by_name('Pooling').update_node_stat(node, mapping_rule)
        return __class__.enabled

    @staticmethod
    def infer(node: Node):
        batch = node.in_node().in_node().in_node().shape[node.batch_dims]
        input_dim_ = node.in_node().in_node().in_node().shape[1]
        num_patches = int(np.ceil(input_dim_ / node.pool_stride))
        num_pools = 1 + int(np.ceil((num_patches - node.window[node.spatial_dims][0]) / node.stride[node.spatial_dims][0]))
        node.out_node(0).shape = int64_array([batch, node.pool_stride, 1, num_pools])
