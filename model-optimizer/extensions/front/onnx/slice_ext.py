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
from mo.ops.slice import Slice


class SliceFrontExtractor(FrontExtractorOp):
    op = 'Slice'
    enabled = True

    @staticmethod
    def extract(node):
        axis = np.array(onnx_attr(node, 'axes', 'ints', default=[]), dtype=np.int64)
        start = np.array(onnx_attr(node, 'starts', 'ints', default=[]), dtype=np.int64)
        end = np.array(onnx_attr(node, 'ends', 'ints', default=[]), dtype=np.int64)

        attrs = {
            'axis': axis if len(axis) != 0 else None,
            'start': start if len(start) != 0 else None,
            'end': end if len(end) != 0 else None,
            'format': 'onnx'
        }

        # update the attributes of the node
        Slice.update_node_stat(node, attrs)
        return __class__.enabled
