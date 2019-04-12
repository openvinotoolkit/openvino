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
from mo.ops.reduce import Reduce


class MaxFrontExtractor(FrontExtractorOp):
    op = 'Max'
    enabled = True

    @staticmethod
    def extract(node):
        data = {
            'reduce_type': 'max',
            'keep_dims': node.pb.attr['keep_dims'].b
        }
        Reduce.update_node_stat(node, data)
        return __class__.enabled
