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

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op


class FakeQuantWithMinMaxVarsExtractor(FrontExtractorOp):
    op = 'FakeQuantWithMinMaxVars'
    enabled = True

    @staticmethod
    def extract(node):
        narrow_range = node.pb.attr['narrow_range'].b
        num_bits = node.pb.attr['num_bits'].i
        levels = 2 ** num_bits - int(narrow_range)

        # we prepare this operation to be converted to FakeQuantize op,
        # but input reconnection is needed, so we don't set infer function and type attribute
        Op.update_node_stat(node, {'op': 'FakeQuantWithMinMaxVars', 'levels': levels,
                                   'narrow_range': narrow_range, 'num_bits': num_bits})

        return __class__.enabled
