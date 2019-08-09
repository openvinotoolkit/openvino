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

from mo.front.common.partial_infer.matmul import tf_matmul_infer


def tf_matmul_ext(pb):
    return {
        'transpose_a': pb.attr['transpose_a'].b,
        'transpose_b': pb.attr['transpose_b'].b,
        'infer': tf_matmul_infer
    }


def tf_batchmatmul_ext(pb):
    adj_x = pb.attr['adj_x'].b
    adj_y = pb.attr['adj_y'].b
    return {
        'op': 'BatchMatMul',
        'type': 'Gemm',
        'transpose_a': adj_x,
        'transpose_b': adj_y,
        'infer': tf_matmul_infer
    }
