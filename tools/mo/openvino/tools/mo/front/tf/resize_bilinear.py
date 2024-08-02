# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.TFResize import TFResize
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor


class ResizeBilinearFrontExtractor(FrontExtractorOp):
    op = 'ResizeBilinear'
    enabled = True

    @classmethod
    def extract(cls, node):
        align_corners = False
        if 'align_corners' in node.pb.attr:
            align_corners = node.pb.attr['align_corners'].b

        half_pixel_centers = False
        if 'half_pixel_centers' in node.pb.attr:
            half_pixel_centers = node.pb.attr['half_pixel_centers'].b

        attrs = {
            'align_corners': align_corners,
            'half_pixel_centers': half_pixel_centers,
            'mode': 'linear',
            'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
        }
        TFResize.update_node_stat(node, attrs)
        return cls.enabled
