"""
 Copyright (c) 2018-2019 Intel Corporation

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

from extensions.ops.constant_fill import ConstantFill
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ConstantFillFrontExtractor(FrontExtractorOp):
    op = 'ConstantFill'
    enabled = True

    @staticmethod
    def extract(node):

        value = onnx_attr(node, 'value', 'f', default=float(0.0))
        input_as_shape = onnx_attr(node, 'input_as_shape', 'i')
        extra_shape = onnx_attr(node, 'extra_shape', 'ints')
        shape = onnx_attr(node, 'shape', 'ints')
        dtype = onnx_attr(node, 'dtype', 'i', 1)

        assert input_as_shape
        assert extra_shape is None
        assert shape is None
        assert dtype == 1

        attrs = {
            'fill_value': value,
            'input_as_shape': input_as_shape,
        }

        ConstantFill.update_node_stat(node, attrs)
        return __class__.enabled
