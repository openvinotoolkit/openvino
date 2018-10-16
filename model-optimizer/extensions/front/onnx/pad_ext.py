"""
 Copyright (c) 2018 Intel Corporation

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

import logging as log

from mo.ops.pad import Pad
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.utils.error import Error

import numpy as np


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @staticmethod
    def extract(node):
        mode = onnx_attr(node, 'mode', 's', default='constant', dst_type=lambda x: x.decode())
        pads = onnx_attr(node, 'pads', 'ints', dst_type=lambda x: np.array(x, dtype=np.int64))
        value = onnx_attr(node, 'value', 'f', default=0)

        assert pads is not None

        if mode.lower() != 'constant':
            log.error('Pad.mode != constant for node {}. It is not supported. '
                'Model conversion is not aborted but the final IR will be not correct.'.format(node.name))

        if value != 0:
            log.error('Pad.value == {} != 0 for node {}. It is not supported. '
                'MOdel conversion is not aborted but the final IR will be not correct.'.format(value, node.name))

        # MO Pad op and ONNX Pad op have different format for pads values
        # MO Pad has Dx2 where D is the total number of dimensions
        # ONNX Pad pads flat layout, so
        # need to reshape and transpose

        pads = pads.reshape([2,-1])
        pads = np.transpose(pads)

        Pad.update_node_stat(node, {'mode': mode, 'pads': pads, 'fill_value': value})
        return __class__.enabled
