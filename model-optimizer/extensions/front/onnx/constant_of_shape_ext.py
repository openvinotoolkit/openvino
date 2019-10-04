"""
 Copyright (c) 2019 Intel Corporation

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
import numpy as np
from onnx import numpy_helper

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.constant_of_shape import ConstantOfShape


class ConstantOfShapeExtractor(FrontExtractorOp):
    op = 'ConstantOfShape'
    enabled = True

    @staticmethod
    def extract(node):
        fill_value = onnx_attr(node, 'value', 't', default=np.array([0.0]), dst_type=lambda x: numpy_helper.to_array(x))

        ConstantOfShape.update_node_stat(node, {'fill_value': fill_value})
        return __class__.enabled
