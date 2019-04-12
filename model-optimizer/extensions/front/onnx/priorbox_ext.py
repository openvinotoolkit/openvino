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
import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op
from mo.front.onnx.extractors.utils import onnx_attr


class PriorBoxFrontExtractor(FrontExtractorOp):
    op = 'PriorBox'
    enabled = True

    @staticmethod
    def extract(node):
        variance = onnx_attr(node, 'variance', 'floats', default=[], dst_type=lambda x: np.array(x, dtype=np.float32))
        if len(variance) == 0:
            variance = [0.1]

        update_attrs = {
            'aspect_ratio': onnx_attr(node, 'aspect_ratio', 'floats', dst_type=lambda x: np.array(x, dtype=np.float32)),
            'min_size': onnx_attr(node, 'min_size', 'floats', dst_type=lambda x: np.array(x, dtype=np.float32)),
            'max_size': onnx_attr(node, 'max_size', 'floats', dst_type=lambda x: np.array(x, dtype=np.float32)),
            'flip': onnx_attr(node, 'flip', 'i', default=0),
            'clip': onnx_attr(node, 'clip', 'i', default=0),
            'variance': list(variance),
            'img_size': onnx_attr(node, 'img_size', 'i', default=0),
            'img_h': onnx_attr(node, 'img_h', 'i', default=0),
            'img_w': onnx_attr(node, 'img_w', 'i', default=0),
            'step': onnx_attr(node, 'step', 'f', default=0.0),
            'step_h': onnx_attr(node, 'step_h', 'f', default=0.0),
            'step_w': onnx_attr(node, 'step_w', 'f', default=0.0),
            'offset': onnx_attr(node, 'offset', 'f', default=0.0),
        }

        # update the attributes of the node
        Op.get_op_class_by_name(__class__.op).update_node_stat(node, update_attrs)
        return __class__.enabled
