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

import math

import numpy as np

from extensions.ops.upsample import UpsampleOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.utils.error import Error


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

        if scales is not None:
            if scales.shape != (4,):
                raise Error(
                    'Upsample scales attribute is wrong for node {}. Only 4D scales are supported.',
                    node.name
                )
            if math.fabs(scales[0] - 1) > 1e-5 or math.fabs(scales[1] - 1) > 1e-5:
                raise Error(
                    'Upsampling of batch and feature dimentions is not supported for node {}.',
                    node.name
                )
            height_scale = scales[2]
            width_scale = scales[3]

        if (width_scale is None or height_scale is None) and len(node.in_nodes()) != 2:
            raise Error(
                'One/both of widths_scale = {} and height_scale = {} is not defined for Upsample node {}.',
                width_scale,
                height_scale,
                node.name
            )

        UpsampleOp.update_node_stat(node, {'mode': mode, 'height_scale': height_scale,
                                           'width_scale': width_scale})
        return __class__.enabled
