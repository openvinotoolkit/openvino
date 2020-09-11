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

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, get_onnx_opset_version
from mo.ops.pad import AttributedPad, ONNXPad


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @classmethod
    def extract(cls, node):
        mode = onnx_attr(node, 'mode', 's', default='constant', dst_type=lambda x: x.decode())
        if get_onnx_opset_version(node) < 11:
            pads = onnx_attr(node, 'pads', 'ints', dst_type=lambda x: np.array(x, dtype=np.int64))
            value = onnx_attr(node, 'value', 'f', default=0.)

            assert pads is not None

            # MO Pad op and ONNX Pad op have different format for pads values
            # MO Pad has Dx2 where D is the total number of dimensions
            # ONNX Pad pads flat layout, so need to reshape and transpose

            pads = np.transpose(pads.reshape([2, -1]))

            AttributedPad.update_node_stat(node, {'mode': mode, 'pads': pads, 'fill_value': value})
        else:
            ONNXPad.update_node_stat(node, {'mode': mode})
        return cls.enabled
