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
from mo.ops.expand_dims import ExpandDims


class UnsqueezeFrontExtractor(FrontExtractorOp):
    """
    Convert Unsqueeze layer to ExpandDims because the ExpandDims layer has fixed attribute with dimensions to unsqueeze.
    """
    op = 'Unsqueeze'
    enabled = True

    @staticmethod
    def extract(node):
        axis = np.array(onnx_attr(node, 'axes', 'ints', default=[]), dtype=np.int64)

        ExpandDims.update_node_stat(node, {'expand_axis': axis})
        return __class__.enabled
