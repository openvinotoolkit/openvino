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

from extensions.ops.reverse_sequence import ReverseSequence
from mo.front.extractor import FrontExtractorOp


class ReverseSequenceFrontExtractor(FrontExtractorOp):
    op = 'ReverseSequence'
    enabled = True

    @staticmethod
    def extract(node):
        if node.has_valid('seq_dim'):
            return

        ReverseSequence.update_node_stat(node, {
            'seq_axis': node.pb.attr['seq_dim'].i,
            'batch_axis': node.pb.attr['batch_dim'].i,
        })
        return __class__.enabled
