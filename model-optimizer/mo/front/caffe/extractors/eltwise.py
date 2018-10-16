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

from mo.front.common.partial_infer.eltwise import eltwise_infer


def eltwise_ext(pl, ml):
    mul_elt_lambda = lambda node: eltwise_infer(node, lambda a, b: a * b)
    sum_elt_lambda = lambda node: eltwise_infer(node, lambda a, b: a + b)
    max_elt_lambda = lambda node: eltwise_infer(node, lambda a, b: np.maximum(a, b))

    param = pl.eltwise_param

    attr_mul = {
        'type': 'Eltwise',
        'op': 'Mul',
        'operation': 'mul',
        'infer': mul_elt_lambda
    }

    attr_sum = {
        'type': 'Eltwise',
        'op': 'Add',
        'coeff': ','.join(str(x) for x in param.coeff),
        'operation': 'sum',
        'infer': sum_elt_lambda
    }

    attr_max = {
        'type': 'Eltwise',
        'op': 'Max',
        'operation': 'max',
        'infer': max_elt_lambda
    }

    eltwise_caffe_map = {
        0: attr_mul,
        1: attr_sum,
        2: attr_max
    }

    operation = int(param.operation)
    if operation in eltwise_caffe_map:
        return eltwise_caffe_map[operation]
    else:
        raise Exception('Unsupported type of operation in Eltwise layer: ' + pl.name)
