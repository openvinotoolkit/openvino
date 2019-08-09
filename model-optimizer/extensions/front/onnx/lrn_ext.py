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

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.lrn import LRN


class LRNFrontExtractor(FrontExtractorOp):
    op = 'LRN'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = {
            'alpha': onnx_attr(node, 'alpha', 'f', 1e-4),
            'beta': onnx_attr(node, 'beta', 'f', 0.75),
            'bias': onnx_attr(node, 'bias', 'f', 1.0),
            'local_size': onnx_attr(node, 'size', 'i', None),
        }
        # TODO To be aligned with the specification, LRN should have axes input instead of old
        # region attributes. This extra input should be build in a separate transformation.
        LRN.update_node_stat(node, attrs)
        return __class__.enabled
