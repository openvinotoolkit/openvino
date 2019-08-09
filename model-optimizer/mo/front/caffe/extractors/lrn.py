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

from mo.front.common.partial_infer.elemental import copy_shape_infer


def lrn_ext(pb_layer, pb_model):
    param = pb_layer.lrn_param
    region = 'across'
    if param.norm_region == 1:
        region = 'same'
    return {
        'type': 'LRN',
        'op': 'LRN',
        'alpha': param.alpha,
        'beta': param.beta,
        'local_size': param.local_size,
        'region': region,
        'bias': 1,
        'infer': copy_shape_infer
    }
