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
from mo.front.common.partial_infer.elemental import single_output_infer


class LoopCondFrontExtractor(FrontExtractorOp):
    op = 'LoopCond'
    enabled = True

    @staticmethod
    def extract(node):
        node['infer'] = lambda node: single_output_infer(
            node,
            lambda node: node.in_node(0).shape,
            lambda node: node.in_node(0).value
        )
        return __class__.enabled
