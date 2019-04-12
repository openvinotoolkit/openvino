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

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class DepthToSpaceFrontExtractor(FrontExtractorOp):
    op = 'DepthToSpace'
    enabled = True

    @staticmethod
    def extract(node):
        # update the attributes of the node
        block_size = node.pb.attr['block_size'].i
        data_format = node.pb.attr['data_format'].s.decode('utf-8')
        Op.get_op_class_by_name(__class__.op).update_node_stat(node,
                                                               {'block_size': block_size, 'data_format': data_format})
        return __class__.enabled
