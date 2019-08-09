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

import numpy as np

from mo.front.common.partial_infer.utils import int64_array
from mo.ops.op import PermuteAttrs
from mo.utils.error import Error


def tf_matmul_infer(node):
    assert (len(node.in_nodes()) == 2)

    shapes = [node.in_node(i).shape.copy() for i in range(2)]
    log.debug('matmul shapes: {}'.format(shapes))
    if any(s is None or len(s) < 2 for s in shapes):
        log.error("MatMul wasn't able to infer shape")
        return

    if node.transpose_a:
        perm = np.array(range(len(node.in_node(0).shape)), dtype=np.int64)
        perm[-1], perm[-2] = perm[-2], perm[-1]
        inv = PermuteAttrs.get_inverse_permutation(perm)
        permutation = PermuteAttrs.Permutation(perm=perm, inv=int64_array(inv))
        PermuteAttrs.set_permutation(node.in_node(0), node, permutation)
        shapes[0] = shapes[0][perm]

    if node.transpose_b:
        perm = np.array(range(len(node.in_node(1).shape)), dtype=np.int64)
        perm[-1], perm[-2] = perm[-2], perm[-1]
        inv = PermuteAttrs.get_inverse_permutation(perm)
        permutation = PermuteAttrs.Permutation(perm=perm, inv=int64_array(inv))
        PermuteAttrs.set_permutation(node.in_node(1), node, permutation)
        shapes[1] = shapes[1][perm]

    if any(shapes[0][:-2] != shapes[1][:-2]) or shapes[0][-1] != shapes[1][-2]:
        log.error("MatMul wasn't able to infer shape because input dimensions are not compatible")
        return

    shape_tuple = (np.array(shapes[0][:-1], dtype=np.int64), np.array([shapes[1][-1]], dtype=np.int64))
    if len(shapes[0]) > 2:
        # TODO Investigate case when MatMul have inputs with not matching output dimensions
        # It looks to be a practical case and if we add outer dimensions of the first argument
        # it will lead to incorrect model sometimes. TF documentation is unclear.
        log.warning('Ignored outer dimensions of input tensor for MatMul node: {}'.format(node.name))
        # shape_tuple = (shapes[0][:-2], *shape_tuple)

    log.debug('shape_tuple: {}'.format(shape_tuple))
    node.out_node().shape = np.concatenate(shape_tuple)
    node['channel_dims'] = node.out_node().shape.size - 1
    log.debug('matmul shape: {}'.format(node.out_node().shape))



def onnx_gemm_infer(node):
    assert (len(node.in_nodes()) == 3)
    shapeA = node.in_node(0).shape
    shapeB = node.in_node(1).shape
    shapeC = node.in_node(2).shape

    assert shapeA.size >= 2 and shapeB.size == 2 and shapeC.size in [1, 2]

    if shapeA.size > 2 and node.transpose_a:
        raise Error(
            'ONNX Gemm operation do not support {} dimensional input with set transA key'.format(shapeA.size))

    # apply transposes and broadcasts
    if node.transpose_a:
        shapeA = shapeA[[1, 0]]
    if node.transpose_b:
        shapeB = shapeB[[1, 0]]
    if node.broadcast_c and shapeC.size == 1:
        shapeC = np.array([shapeA[0], shapeB[1]])

    node.out_node().shape = shapeC
    return
