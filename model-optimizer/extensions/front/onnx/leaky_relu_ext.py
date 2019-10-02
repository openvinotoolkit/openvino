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
from extensions.ops.activation_ops import LeakyReLU, ReLU
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class LeakyReLUFrontExtractor(FrontExtractorOp):
    op = 'LeakyRelu'
    enabled = True

    @staticmethod
    def extract(node):
        negative_slope = onnx_attr(node, 'alpha', 'f', default=1.0)
        if negative_slope == 0:
            ReLU.update_node_stat(node)
        else:
            LeakyReLU.update_node_stat(node, {'negative_slope': negative_slope})
        return __class__.enabled
