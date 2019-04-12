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

import logging as log

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.crop import Crop


class CropFrontExtractor(FrontExtractorOp):
    op = 'Crop'
    enabled = True

    @staticmethod
    def extract(node):
        # borders: leftBorder, topBorder, rightBorder, bottomBordes
        borders = onnx_attr(node, 'border', 'ints', default=None, dst_type=int64_array)
        scale = onnx_attr(node, 'scale', 'ints', default=None, dst_type=int64_array)

        # Crop reference: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Crop
        if len(borders) != 4:
            log.error('ONNX Crop layer {} should take exactly 4 borders instead of {}'.format(node.name, len(borders)))
            return False

        attrs = {'axis': int64_array([2, 3])}
        if scale is not None:
            attrs.update({
                'dim': scale,
                'offset': int64_array([borders[1], borders[0]])
            })
        else:
            attrs.update({
                'crop_begin': int64_array([borders[1], borders[0]]),
                'crop_end': int64_array([borders[3], borders[2]])
            })

        Crop.update_node_stat(node, attrs)
        return CropFrontExtractor.enabled
