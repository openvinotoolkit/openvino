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

from mo.front.common.partial_infer.elemental import single_output_infer
from mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from mo.ops.op import PermuteAttrs


def tf_placeholder_ext(pb):
    return {
        'data_type': tf_dtype_extractor(pb.attr["dtype"].type),
        'shape': tf_tensor_shape(pb.attr["shape"].shape),
        'type': 'Input',
        'infer': lambda node: single_output_infer(node, lambda n: n.shape),
        'permute_attrs': PermuteAttrs().update_attrs(attrs=[('shape', 'output:0')])
    }
