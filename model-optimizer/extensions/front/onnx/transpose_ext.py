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

from extensions.ops.transpose import Transpose
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class TransposeFrontExtractor(FrontExtractorOp):
    op = 'Transpose'
    enabled = True

    @staticmethod
    def extract(node):
        # In case of undefined 'perm' attribute, Transpose operation in ONNX reverse the dimensions
        order = onnx_attr(node, 'perm', 'ints', default=None)
        attrs = {
            'order': int64_array(order) if order is not None else None,
            'reverse_order': order is None
        }
        Transpose.update_node_stat(node, attrs)
        return __class__.enabled