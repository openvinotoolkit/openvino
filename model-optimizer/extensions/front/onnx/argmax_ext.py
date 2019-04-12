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

from extensions.ops.argmax import ArgMaxOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr

class ArgMaxFrontExtractor(FrontExtractorOp):
    op = 'ArgMax'
    enabled = True

    @staticmethod
    def extract(node):
        keepdims = onnx_attr(node, 'keepdims', 'i', default=1)
        axis = onnx_attr(node, 'axis', 'i', default=0)

        attrs = {
            'axis': axis,

            # ONNX ArgMax always computes an index of one maximum value
            'top_k' : 1,
            'out_max_val' : 0,

            # Set attribute to trigger ArgMax replacer in case do not keep the dimension
            'keepdims': keepdims
        }

        ArgMaxOp.update_node_stat(node, attrs)
        return __class__.enabled
