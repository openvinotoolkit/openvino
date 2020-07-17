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

from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import get_onnx_opset_version
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.slice import Slice, AttributedSlice


class SliceFrontExtractor(FrontExtractorOp):
    op = 'Slice'
    enabled = True

    @classmethod
    def extract(cls, node):
        if get_onnx_opset_version(node) < 10:
            axes = int64_array(onnx_attr(node, 'axes', 'ints', default=[]))
            starts = int64_array(onnx_attr(node, 'starts', 'ints', default=[]))
            ends = int64_array(onnx_attr(node, 'ends', 'ints', default=[]))

            attrs = {
                'axes': axes if len(axes) != 0 else None,
                'starts': starts if len(starts) != 0 else None,
                'ends': ends if len(ends) != 0 else None,
            }
            AttributedSlice.update_node_stat(node, attrs)
        else:  # onnx_opset_version >= 10
            Slice.update_node_stat(node)
        return cls.enabled
