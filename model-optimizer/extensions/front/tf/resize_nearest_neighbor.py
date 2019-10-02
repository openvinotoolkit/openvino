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
from extensions.ops.interpolate import Interpolate
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp


class ResizeNearestNeighborFrontExtractor(FrontExtractorOp):
    op = 'ResizeNearestNeighbor'
    enabled = True

    @staticmethod
    def extract(node):
        mapping_rule = {
            'mode': 'nearest',
            'antialias': 0,
            'axes': int64_array([1, 2]),
        }
        Interpolate.update_node_stat(node, mapping_rule)
        return __class__.enabled
