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
from extensions.ops.roialign import ROIAlign
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


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

        ROIAlign.update_node_stat(node, {'pooled_h': output_height, 'pooled_w': output_width,
                                         'sampling_ratio': sampling_ratio, 'spatial_scale': spatial_scale,
                                         'mode': mode})
        return cls.enabled
