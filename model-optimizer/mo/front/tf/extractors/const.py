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

from mo.front.common.partial_infer.const import tf_const_infer
from mo.front.tf.extractors.utils import tf_dtype_extractor, tf_tensor_shape
from mo.front.tf.extractors.utils import tf_tensor_content


def tf_const_ext(pb):
    pb_tensor = pb.attr["value"].tensor
    result = {
        'data_type': tf_dtype_extractor(pb_tensor.dtype),
        'shape': tf_tensor_shape(pb_tensor.tensor_shape),
        'infer': tf_const_infer
    }
    result['value'] = tf_tensor_content(pb_tensor.dtype, result['shape'], pb_tensor)
    log.debug('Constant extractor for node gives shape = {} and value.shape = {}'.format(result['shape'],
                                                                                         result['value'].shape))
    return result
