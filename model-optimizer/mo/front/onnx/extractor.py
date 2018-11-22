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

import logging as log

import numpy as np

from mo.front.onnx.extractors.const import onnx_const_ext
from mo.front.onnx.extractors.constant import onnx_constant_ext
from mo.front.onnx.extractors.eltwise import make_tf_eltwise
from mo.front.onnx.extractors.fused_bn import tf_fused_bn_extractor
from mo.front.onnx.extractors.matmul import onnx_gemm_ext
from mo.front.onnx.extractors.placeholder import onnx_placeholder_ext
from mo.front.onnx.extractors.concat import concat_ext
from mo.front.onnx.extractors.dropout import dropout_ext
from mo.front.onnx.extractors.reshape import onnx_reshape_ext
from mo.front.tf.extractors.softmax import tf_softmax_ext
from mo.graph.graph import Node


def node_pb_arg(pb_extractor: callable):
    return lambda node: pb_extractor(node.pb)


onnx_op_extractors = {
    'BatchNormalization': tf_fused_bn_extractor,
    'Gemm': onnx_gemm_ext,
    'Placeholder': onnx_placeholder_ext,
    'Concat': concat_ext,
    'Const': onnx_const_ext,
    'Constant': onnx_constant_ext,
    'Identity': node_pb_arg(make_tf_eltwise(lambda v: v)),
    'Dropout': dropout_ext,
    'Sum': node_pb_arg(
        make_tf_eltwise(lambda a, b: a + b, attrs={'type': 'Eltwise', 'operation': 'sum', 'can_be_bias': True})),
    'Relu': node_pb_arg(make_tf_eltwise(lambda v: np.maximum(0, v), attrs={'type': 'ReLU'})),  # 0 is an integer
    'Reshape': onnx_reshape_ext,
    'Softmax': node_pb_arg(tf_softmax_ext),
}


def common_onnx_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.id,  # no reliable name for an onnx node, name can be empty, so we use that surrogate built as ID in the loaader
        'op': node.op if node.has_valid('op') else node.pb.op_type,
        'precision': 'FP32'  # TODO use real precision derived from the model
    }


def onnx_op_extractor(node: Node, lowered_keys_map: dict):
    if not node.has_valid('pb'):
        return True, node.graph.node[node.id]

    result = common_onnx_fields(node)
    node.graph.node[node.id].update(result)
    supported = False
    op = result['op'].lower()
    if op in lowered_keys_map:
        op = lowered_keys_map[op]
        assert op in onnx_op_extractors
        attrs = onnx_op_extractors[op](node)
        if attrs:
            result.update(attrs)
            supported = True
    return supported, result
