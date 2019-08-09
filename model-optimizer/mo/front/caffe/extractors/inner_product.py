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

from mo.front.caffe.extractors.utils import weights_biases
from mo.front.common.partial_infer.inner_product import caffe_inner_product


def inner_product_ext(pb_layer, pb_model):
    param = pb_layer.inner_product_param
    attrs = {
        'op': 'MatMul',
        'type': 'MatMul',
        'out-size': param.num_output,
        'layout': 'NCHW',
        'infer': caffe_inner_product
    }
    attrs.update(weights_biases(param.bias_term, pb_model))
    return attrs
