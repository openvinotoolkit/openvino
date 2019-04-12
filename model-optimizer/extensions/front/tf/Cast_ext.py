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

from extensions.ops.Cast import Cast
from mo.front.extractor import FrontExtractorOp
from mo.front.tf.common import tf_data_type_decode


class CastFrontExtractor(FrontExtractorOp):
    op = 'Cast'
    enabled = True

    @staticmethod
    def extract(node):
        cast_dst_type = tf_data_type_decode[node.pb.attr['DstT'].type][0]
        Cast.update_node_stat(node, {'dst_type': cast_dst_type})
        return __class__.enabled
