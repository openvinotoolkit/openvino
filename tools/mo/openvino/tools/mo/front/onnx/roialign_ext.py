# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.roialign import ROIAlign
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version


class ROIAlignExtractor(FrontExtractorOp):
    op = 'ROIAlign'
    enabled = True

    @classmethod
    def extract(cls, node):
        mode = onnx_attr(node, 'mode', 's', default=b'avg').decode()
        output_height = onnx_attr(node, 'output_height', 'i', default=1)
        output_width = onnx_attr(node, 'output_width', 'i', default=1)
        sampling_ratio = onnx_attr(node, 'sampling_ratio', 'i', default=0)
        spatial_scale = onnx_attr(node, 'spatial_scale', 'f', default=1.0)
        onnx_opset_version = get_onnx_opset_version(node)
        if onnx_opset_version >= 16:
            aligned_mode = onnx_attr(node, 'coordinate_transformation_mode', 's', default=b'half_pixel').decode()
            if aligned_mode == "output_half_pixel":
                aligned_mode = "asymmetric"
            ROIAlign.update_node_stat(node, {'pooled_h': output_height, 'pooled_w': output_width,
                                            'sampling_ratio': sampling_ratio, 'spatial_scale': spatial_scale,
                                            'mode': mode, 'aligned_mode': aligned_mode})
        else:
            ROIAlign.update_node_stat(node, {'pooled_h': output_height, 'pooled_w': output_width,
                                            'sampling_ratio': sampling_ratio, 'spatial_scale': spatial_scale,
                                            'mode': mode})
        return cls.enabled
