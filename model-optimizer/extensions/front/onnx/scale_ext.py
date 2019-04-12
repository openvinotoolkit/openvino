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


class ScaleFrontExtractor(FrontExtractorOp):
    op = 'Scale'
    enabled = True

    @staticmethod
    def extract(node):
        scale = onnx_attr(node, 'scale', 'f', default=np.array(1.0), dst_type=lambda x: np.array(x))

        node['scale'] = scale
        node['bias'] = np.array(0)
        node['op'] = 'ImageScaler'

        return __class__.enabled
