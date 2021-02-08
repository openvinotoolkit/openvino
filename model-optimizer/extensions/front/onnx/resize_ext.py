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

from extensions.ops.ONNXResize10 import ONNXResize10
from extensions.ops.ONNXResize11 import ONNXResize11Op
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from mo.graph.graph import Node


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
