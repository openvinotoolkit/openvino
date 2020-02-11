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

from extensions.ops.non_max_suppression import NonMaxSuppression
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class NonMaxSuppressionExtractor(FrontExtractorOp):
    op = 'NonMaxSuppression'
    enabled = True

    @classmethod
    def extract(cls, node):
        encoding_map = {0: 'corner', 1: 'center'}
        center_point_box = onnx_attr(node, 'center_point_box', 'i', default=0)
        NonMaxSuppression.update_node_stat(node, {'sort_result_descending': 0,
                                                  'box_encoding': encoding_map[center_point_box]})
        return cls.enabled
