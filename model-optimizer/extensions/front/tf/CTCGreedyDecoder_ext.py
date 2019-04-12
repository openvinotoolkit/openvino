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
from extensions.ops.ctc_greedy_decoder import CTCGreedyDecoderOp
from mo.front.extractor import FrontExtractorOp


class CTCCGreedyDecoderFrontExtractor(FrontExtractorOp):
    op = 'CTCGreedyDecoder'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = {
            'ctc_merge_repeated': int(node.pb.attr['merge_repeated'].b),
        }
        CTCGreedyDecoderOp.update_node_stat(node, attrs)
        return __class__.enabled
