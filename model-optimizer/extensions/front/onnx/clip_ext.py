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
from mo.ops.clamp import Clamp


class ClipFrontExtractor(FrontExtractorOp):
    op = 'Clip'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = {
            'min': onnx_attr(node, 'min', 'f', -3.4028234663852886e+38),
            'max': onnx_attr(node, 'max', 'f', 3.4028234663852886e+38),
        }
        Clamp.update_node_stat(node, attrs)
        return __class__.enabled
