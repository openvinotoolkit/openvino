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
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp


class ResizeBilinearFrontExtractor(FrontExtractorOp):
    op = 'ResizeBilinear'
    enabled = True

    @classmethod
    def extract(cls, node):
        transformation_mode = 'align_corners' if int(node.pb.attr['align_corners'].b) else 'half_pixel'
        node_attributes = {
            'axes': int64_array([1, 2]),
            'mode': 'linear',
            'antialias': 0,
            'pads_begin': int64_array([0]),
            'pads_end': int64_array([0]),
            'coordinate_transformation_mode': transformation_mode,
            'nearest_mode': 'round_prefer_floor',
            'cube_coeff': -0.75,
            'version': 'opset3'
        }
        Interpolate.update_node_stat(node, node_attributes)
        return cls.enabled
