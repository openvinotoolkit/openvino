"""
 Copyright (c) 2018 Intel Corporation

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

from extensions.ops.resample import ResampleOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.utils.error import Error

import numpy as np


class UpsampleFrontExtractor(FrontExtractorOp):
    op = 'Upsample'
    enabled = True

    @staticmethod
    def extract(node):
        mode = onnx_attr(node, 'mode', 's', default='nearest', dst_type=lambda x: x.decode())
        scales = onnx_attr(node, 'scales', 'floats', dst_type=lambda x: np.array(x, dtype=np.float32))
        width_scale = onnx_attr(node, 'width_scale', 'f')
        height_scale = onnx_attr(node, 'height_scale', 'f')

        supported_modes = ['nearest', 'linear']
        if mode not in supported_modes:
            raise Error(
                'Error decoding Upsample node {}, mode = {} is not in the list of supported modes {}.',
                node.name,
                mode,
                supported_modes
            )

        # TODO: this is a temporary limitation
        if mode != 'nearest':
            raise Error(
                'Upsample mode {} for node {} is not supported. Only nearest is supported.',
                mode,
                node.name
            )

        # TODO: this is a temporary limitation
        if scales is not None:
            raise Error(
                'Upsample scales attribute is defined for node {}. Only scale_width and scale_height are supported.',
                node.name
            )

        if width_scale is None or height_scale is None:
            raise Error(
                'One/both of widths_scale = {} and height_scale = {} is not defined for Upsampe node {}.',
                width_scale,
                height_scale,
                node.name
            )

        if width_scale != height_scale:
            raise Error(
                'Upsample node {} have different widths_scale = {} and height_scale = {}. It is not supported; they should match.',
                node.name,
                width_scale,
                height_scale
            )

        mode_to_resample_type = {'nearest': 'caffe.ResampleParameter.NEAREST'}
        assert mode in mode_to_resample_type
        assert width_scale == height_scale
        assert width_scale is not None
        ResampleOp.update_node_stat(node, {'resample_type': mode_to_resample_type[mode], 'factor': width_scale, 'antialias': 0})
        return __class__.enabled
