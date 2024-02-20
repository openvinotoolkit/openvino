# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ONNXResize10 import ONNXResize10
from openvino.tools.mo.ops.ONNXResize11 import ONNXResize11Op
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from openvino.tools.mo.graph.graph import Node


class ResizeExtractor(FrontExtractorOp):
    op = 'Resize'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        onnx_opset_version = get_onnx_opset_version(node)
        if onnx_opset_version is not None and onnx_opset_version >= 11:
            mode = onnx_attr(node, 'mode', 's', default=b'nearest').decode()
            transformation_mode = onnx_attr(node,
                                            'coordinate_transformation_mode',
                                            's',
                                            default=b'half_pixel').decode()
            nearest_mode = onnx_attr(node, 'nearest_mode', 's', default=b'round_prefer_floor').decode()
            cubic_coeff_a = onnx_attr(node, 'cubic_coeff_a', 'f', default=-0.75)
            attrs = {
                'mode': mode, 'coordinate_transformation_mode': transformation_mode,
                'nearest_mode': nearest_mode, 'cube_coeff': cubic_coeff_a
            }
            ONNXResize11Op.update_node_stat(node, attrs)
        else:
            mode = onnx_attr(node, 'mode', 's', default=b'nearest').decode()
            ONNXResize10.update_node_stat(node, {'mode': mode})
        return cls.enabled
