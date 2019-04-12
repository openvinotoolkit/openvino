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

from extensions.ops.proposal_onnx import ExperimentalDetectronGenerateProposalsSingleImage
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class ExperimentalDetectronGenerateProposalsSingleImageFrontExtractor(FrontExtractorOp):
    op = 'ExperimentalDetectronGenerateProposalsSingleImage'
    enabled = True

    @staticmethod
    def extract(node):
        attrs = dict(min_size=onnx_attr(node, 'min_size', 'f', 0.0),
                     nms_threshold=onnx_attr(node, 'nms_threshold', 'f', 0.7),
                     post_nms_count=onnx_attr(node, 'post_nms_count', 'i', 1000),
                     pre_nms_count=onnx_attr(node, 'pre_nms_count', 'i', 1000)
                     )
        ExperimentalDetectronGenerateProposalsSingleImage.update_node_stat(node, attrs)
        return __class__.enabled
