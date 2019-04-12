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

from extensions.ops.priorgridgenerator_onnx import ExperimentalDetectronPriorGridGenerator
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronPriorGridGeneratorFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronPriorGridGenerator'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = dict(h=onnx_attr(node, 'h', 'i', 0),
                     w=onnx_attr(node, 'w', 'i', 0),
                     stride_x=onnx_attr(node, 'stride_x', 'f', 0),
                     stride_y=onnx_attr(node, 'stride_y', 'f', 0),
                     flatten=onnx_attr(node, 'flatten', 'i', 1)
                     )
        ExperimentalDetectronPriorGridGenerator.update_node_stat(node, attrs)
        return __class__.enabled
