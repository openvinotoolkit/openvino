"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.back.replacement import BackReplacementPattern
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Graph


class StridedSliceMasksNormalizer(BackReplacementPattern):
    enabled = True
    force_clean_up = True

    def run_after(self):
        from extensions.back.ConvolutionNormalizer import DeconvolutionNormalizer
        from extensions.back.CropToStridedSlice import CropToStridedSlice
        return [CropToStridedSlice, DeconvolutionNormalizer]

    def find_and_replace_pattern(self, graph: Graph):
        for node in graph.get_op_nodes(type='StridedSlice'):
            assert node.has_valid('begin_mask')
            assert node.has_valid('end_mask')
            node.begin_mask = int64_array([1 - i for i in node.begin_mask])
            node.end_mask = int64_array([1 - i for i in node.end_mask])
