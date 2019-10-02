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

from mo.front.common.partial_infer.eltwise import eltwise_infer
from mo.front.tf.extractors.utils import tf_dtype_extractor


def tf_eltwise_ext(pb, op=None, attrs=None):
    """
    Generic eltwise extractor that supports n-ary operations.
    It supports reasonable broadcast semantics from TF/NumPy
    """
    res = {
        'data_type': tf_dtype_extractor(pb.attr["T"].type),
        'infer': lambda node: eltwise_infer(node, op)
    }
    if attrs is not None:
        res.update(attrs)
    return res


def make_tf_eltwise(op, attrs=None):
    return lambda node: tf_eltwise_ext(node, op, attrs)
