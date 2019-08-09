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

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.group_norm import GroupNorm


class GroupNormExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronGroupNorm'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = {
            'eps': np.array(onnx_attr(node, 'eps', 'f', default=1e-6), dtype=np.float),
            'num_groups': np.array(onnx_attr(node, 'num_groups', 'i', default=1), dtype=np.int64),
        }
        GroupNorm.update_node_stat(node, attrs)
        return __class__.enabled
