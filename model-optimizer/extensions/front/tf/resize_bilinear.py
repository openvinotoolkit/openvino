"""
 Copyright (C) 2018-2020 Intel Corporation

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

from extensions.ops.TFResize import TFResize
from mo.front.extractor import FrontExtractorOp


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
            'mode': 'linear'
        }
        TFResize.update_node_stat(node, attrs)
        return cls.enabled
