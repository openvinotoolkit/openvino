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
from mo.ops.op import Op

from mo.front.onnx.extractors.utils import onnx_attr


class ImageScalerFrontExtractor(FrontExtractorOp):
    op = 'ImageScaler'
    enabled = True

    @staticmethod
    def extract(node):
        dst_type = lambda x: np.array(x)

        scale = onnx_attr(node, 'scale', 'f', default=np.array(1.0), dst_type=dst_type)
        bias = onnx_attr(node, 'bias', 'floats', default=None, dst_type=dst_type)

        # Expand dims for bias in case if it is not scalar
        if bias.ndim != 0:
            broadcast_dims_cnt = 2 if node.graph.graph['layout'] == 'NCHW' else 0
            for idx in range(broadcast_dims_cnt):
                bias = np.expand_dims(bias, axis=-1)

        node['scale'] = scale
        node['bias'] = bias

        return __class__.enabled