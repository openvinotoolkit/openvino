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

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class AffineFrontExtractor(FrontExtractorOp):
    # Affine operation will be transformed to ImageScalar and further will be converted to Mul->Add seq
    op = 'Affine'
    enabled = True

    @staticmethod
    def extract(node):
        dst_type = lambda x: np.array(x)

        scale = onnx_attr(node, 'alpha', 'f', default=None, dst_type=dst_type)
        bias = onnx_attr(node, 'beta', 'f', default=None, dst_type=dst_type)

        node['scale'] = scale
        node['bias'] = bias
        node['op'] = 'ImageScaler'

        return __class__.enabled
