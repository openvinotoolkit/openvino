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

import ast
import numpy as np

from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.const import Const


class ZerosFrontExtractor(FrontExtractorOp):
    op = '_zeros'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        shape = list(attrs.tuple('shape', int, None))
        zero_shapes = []
        for i, s in enumerate(shape):
            if s == 0:
                shape[i] = 1
                zero_shapes.append(i)

        update_attrs = {
            'shape': np.ndarray(shape),
            'value': np.zeros(shape),
            'zero_shapes': zero_shapes
        }

        # update the attributes of the node
        Const.update_node_stat(node, update_attrs)
        return __class__.enabled
