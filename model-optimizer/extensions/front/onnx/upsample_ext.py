# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

from extensions.ops.ONNXResize10 import ONNXResize10
from extensions.ops.upsample import UpsampleOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from mo.utils.error import Error


class UpsampleFrontExtractor(FrontExtractorOp):
    op = 'Upsample'
    enabled = True

    @classmethod
    def extract(cls, node):
        onnx_opset_version = get_onnx_opset_version(node)
        if onnx_opset_version is not None and onnx_opset_version >= 9:
            mode = onnx_attr(node, 'mode', 's', default='nearest', dst_type=lambda x: x.decode())
            ONNXResize10.update_node_stat(node, {'mode': mode})
        else:
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
                        'Upsampling of batch and feature dimensions is not supported for node {}.',
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
        return cls.enabled
