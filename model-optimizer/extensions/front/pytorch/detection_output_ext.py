"""
 Copyright (C) 2018-2020 Intel Corporation

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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from extensions.ops.DetectionOutput import DetectionOutput
from mo.utils.error import Error


class DetectionOutputExtractor(FrontExtractorOp):
    op = 'DetectionOutput'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'variance_encoded_in_target': int(node.module.variance_encoded_in_target),
            'nms_threshold': node.module.nms_threshold,
            'confidence_threshold': node.module.confidence_threshold,
            'top_k': node.module.top_k,
            'keep_top_k': node.module.keep_top_k,
            'code_type': node.module.code_type,
        }
        DetectionOutput.update_node_stat(node, attrs)
        return cls.enabled
