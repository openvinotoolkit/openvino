# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.ops.crop import Crop


class CropFrontExtractor(FrontExtractorOp):
    op = 'Crop'
    enabled = True

    @classmethod
    def extract(cls, node):
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
