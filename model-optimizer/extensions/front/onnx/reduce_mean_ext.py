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

from extensions.ops.ReduceOps import ReduceMean
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node


class ReduceMeanFrontExtractor(FrontExtractorOp):
    op = 'ReduceMean'
    enabled = True

    @staticmethod
    def extract(node: Node):
        axis = onnx_attr(node, 'axes', 'ints', default=None, dst_type=lambda x: int64_array(x))
        keep_dims = onnx_attr(node, 'keepdims', 'i', default=True)
        ReduceMean.update_node_stat(node, {'axis': axis, 'keep_dims': keep_dims})
        return __class__.enabled
