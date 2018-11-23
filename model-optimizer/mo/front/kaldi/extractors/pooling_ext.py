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

from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class PoolingFrontExtractor(FrontExtractorOp):
    op = 'pooling'
    enabled = True

    @staticmethod
    def extract(node):
        mapping_rule = {
            'window': int64_array([1, 1, 1, node.pb.kernel]),
            'stride': int64_array([1, 1, node.pb.stride, node.pb.stride]),
            'pool_stride': node.pb.pool_stride,
            'pad': int64_array([[0, 0], [0, 0], [0, 0], [0, 0]]),
            'pad_spatial_shape': int64_array([[0, 0], [0, 0]]),
        }
        mapping_rule.update(layout_attrs())
        Op.get_op_class_by_name('Pooling').update_node_stat(node, mapping_rule)
        return __class__.enabled
