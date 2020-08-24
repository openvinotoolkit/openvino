"""
 Copyright (C) 2020 Intel Corporation

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
from extensions.ops.ctc_loss import CTCLoss
from mo.front.extractor import FrontExtractorOp


class CTCLossFrontExtractor(FrontExtractorOp):
    op = 'CTCLoss'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'ctc_merge_repeated': node.pb.attr['ctc_merge_repeated'].b,
            'preprocess_collapse_repeated': node.pb.attr['preprocess_collapse_repeated'].b,
            # unique is always false for CTCLoss V1
            'unique': False
        }
        CTCLoss.update_node_stat(node, attrs)
        return cls.enabled
