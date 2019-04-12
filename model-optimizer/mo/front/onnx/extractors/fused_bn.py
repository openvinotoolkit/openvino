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
import logging as log

from mo.front.common.partial_infer.elemental import copy_shape_infer
from mo.front.onnx.extractors.utils import onnx_attr


def tf_fused_bn_extractor(node):
    pb = node.pb
    # This statement covers different opset versions
    if onnx_attr(node, 'is_test', 'i', None) == 0:
        log.error('FusedBatchNorm doesn\'t support is_test=False')
        return None

    return {
        'data_format': 'NCHW',
        'eps': onnx_attr(node, 'epsilon', 'f', 1e-5),
        'infer': copy_shape_infer,
    }
