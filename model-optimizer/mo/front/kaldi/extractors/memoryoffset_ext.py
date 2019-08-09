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
from mo.ops.memoryoffset import MemoryOffset


class MemoryOffsetFrontExtractor(FrontExtractorOp):
    op = 'MemoryOffset'
    enabled = True

    @staticmethod
    def extract(node):
        pb = node.parameters
        mapping_rule = {
            'pair_name': pb['pair_name'],
            't': pb['t'],
            'has_default': pb['has_default'],
            'splitted': False,
        }
        if 'element_size' in pb:
            mapping_rule['element_size'] = pb['element_size']

        MemoryOffset.update_node_stat(node, mapping_rule)
        return __class__.enabled
