# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.TFResize import TFResize
from openvino.tools.mo.front.extractor import FrontExtractorOp


class ResizeNearestNeighborFrontExtractor(FrontExtractorOp):
    op = 'ResizeNearestNeighbor'
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
            'mode': 'nearest'
        }
        TFResize.update_node_stat(node, attrs)
        return cls.enabled
