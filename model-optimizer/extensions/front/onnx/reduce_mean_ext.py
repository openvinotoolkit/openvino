"""
 Copyright (c) 2018 Intel Corporation

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
import numpy as np
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.graph.graph import Node
from mo.ops.reduce import Reduce


class ReduceMeanFrontExtractor(FrontExtractorOp):
    op = 'ReduceMean'
    enabled = True

    @staticmethod
    def extract(node: Node):
        axis = onnx_attr(node, 'axes', 'ints', default=None, dst_type= lambda x: np.array(x, dtype=np.int64))
        keep_dims = onnx_attr(node, 'keepdims', 'i', default=True)
        Reduce.update_node_stat(node, {'axis': axis, 'keep_dims': keep_dims, 'reduce_type': 'mean'})
        return __class__.enabled
