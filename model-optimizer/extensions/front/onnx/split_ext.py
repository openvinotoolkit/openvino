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
import numpy as np

from extensions.ops.split import AttributedVariadicSplit, AttributedSplit
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr, onnx_get_num_outputs


class SplitFrontExtractor(FrontExtractorOp):
    op = 'Split'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=0, dst_type=np.int64)
        size_splits = onnx_attr(node, 'split', 'ints', default=None, dst_type=int64_array)
        if size_splits is None:
            AttributedSplit.update_node_stat(node, {
                'axis': axis,
                'num_splits': onnx_get_num_outputs(node),
            })
        else:
            AttributedVariadicSplit.update_node_stat(node, {
                'axis': axis,
                'size_splits': size_splits,
            })
        return cls.enabled
