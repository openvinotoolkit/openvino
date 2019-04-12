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

from mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from mo.front.extractor import FrontExtractorOp
from mo.ops.pad import Pad


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        pads = np.array(list(attrs.tuple('pad_width', int, None)))
        pads = pads.reshape([-1, 2])
        value = attrs.float('constant_value', 0.0)

        node_attrs = {
            'pads': pads,
            'mode': attrs.str('mode', None),
            'fill_value': value,
        }

        Pad.update_node_stat(node, node_attrs)
        return __class__.enabled
