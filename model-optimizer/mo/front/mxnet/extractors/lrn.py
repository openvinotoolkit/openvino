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


def lrn_ext(attrs):
    alpha = attrs.float("alpha", 0.0001)
    beta = attrs.float("beta", 0.75)
    knorm = attrs.float("knorm", 2.0)
    local_size = attrs.int("nsize", None)

    node_attrs = {
        'type': 'LRN',
        'alpha': alpha,
        'beta': beta,
        'knorm': knorm,
        'local_size': local_size,
        'bias': 1,
        'infer': copy_shape_infer
    }
    return node_attrs
