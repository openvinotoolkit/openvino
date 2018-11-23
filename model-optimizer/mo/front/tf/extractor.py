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

from mo.front.common.partial_infer.split import tf_split_infer
from mo.front.tf.extractors.bias_add import tf_bias_add_ext
from mo.front.tf.extractors.concat import tf_concat_ext
from mo.front.tf.extractors.const import tf_const_ext
from mo.front.tf.extractors.eltwise import make_tf_eltwise
from mo.front.tf.extractors.expand_dims import tf_expand_dims_ext
from mo.front.tf.extractors.fused_bn import tf_fused_bn_extractor
from mo.front.tf.extractors.lrn import tf_lrn_ext
from mo.front.tf.extractors.matmul import tf_matmul_ext
from mo.front.tf.extractors.mean import tf_mean_ext
from mo.front.tf.extractors.native_tf import native_tf_node_extractor
from mo.front.tf.extractors.pack import tf_pack_ext
from mo.front.tf.extractors.placeholder import tf_placeholder_ext
from mo.front.tf.extractors.prod import tf_reduce_prod_ext
from mo.front.tf.extractors.random_uniform import tf_random_uniform_ext
from mo.front.tf.extractors.range import tf_range_ext
from mo.front.tf.extractors.reshape import tf_reshape_ext
from mo.front.tf.extractors.shape import tf_shape_ext
from mo.front.tf.extractors.softmax import tf_softmax_ext
from mo.front.tf.extractors.space_to_batch import tf_space_to_batch_ext, tf_batch_to_space_ext
from mo.front.tf.extractors.split import tf_split_ext
from mo.front.tf.extractors.squeeze import tf_squeeze_ext
from mo.front.tf.extractors.strided_slice import tf_strided_slice_ext
from mo.front.tf.extractors.sum import tf_sum_ext
from mo.front.tf.extractors.transpose import tf_transpose_ext
from mo.front.tf.extractors.unpack import tf_unpack_ext
from mo.front.tf.extractors.utils import get_tf_node_port
from mo.graph.graph import Node


def get_tf_edges(node: Node):
    """
    By TF/NX node find all inputs and return list of all edges.
    Edge direction represents data flow (from source op to this node).
    So the resulting list contains all input edges for a given node.
    Edge attributes: 'in' is index of input port for a given node, 'out' is an index
    of output port of some other node that produces input data for this node.
    """
    edge_list = []
    for in_port, src_node_id in enumerate(node.pb.input):
        src_node, src_port = get_tf_node_port(src_node_id)
        cf_flag = False
        if src_node[0] == '^':
            src_node = src_node[1:]
            cf_flag = True
        edge = (src_node, node.id, {
            'in': in_port,
            'out': src_port,
            'fw_tensor_debug_info': [(src_node_id, src_port)],  # debug anchor for a framework tensor name and port
            'in_attrs': ['in', 'control_flow_edge', 'permutation'],
            'out_attrs': ['out', 'permutation'],
            'data_attrs': ['fw_tensor_debug_info'],
            'control_flow_edge': cf_flag
        })
        edge_list.append(edge)
    return edge_list


def node_pb_arg(pb_extractor: callable):
    return lambda node: pb_extractor(node.pb)


tf_op_extractors = {
    'TFCustomSubgraphCall': node_pb_arg(lambda pb: None),
    'Transpose': node_pb_arg(tf_transpose_ext),
    'LRN': node_pb_arg(tf_lrn_ext),
    'Split': node_pb_arg(lambda pb: tf_split_ext(pb, tf_split_infer)),
    'FusedBatchNorm': node_pb_arg(tf_fused_bn_extractor),
    'Relu6': node_pb_arg(
        make_tf_eltwise(lambda a: np.maximum(0, np.minimum(a, 6)), attrs={'type': 'Clamp', 'min': 0, 'max': 6})),
    'ExpandDims': node_pb_arg(tf_expand_dims_ext),
    'ConcatV2': node_pb_arg(tf_concat_ext),
    'MatMul': node_pb_arg(tf_matmul_ext),
    'Pack': node_pb_arg(tf_pack_ext),
    'Unpack': node_pb_arg(tf_unpack_ext),
    'StridedSlice': node_pb_arg(tf_strided_slice_ext),
    'Prod': node_pb_arg(tf_reduce_prod_ext),
    'Const': node_pb_arg(tf_const_ext),
    'Placeholder': node_pb_arg(tf_placeholder_ext),
    'Identity': node_pb_arg(make_tf_eltwise(lambda v: v)),
    'Add': node_pb_arg(
        make_tf_eltwise(lambda a, b: a + b, attrs={'type': 'Eltwise', 'operation': 'sum', 'can_be_bias': True})),
    'Mul': node_pb_arg(make_tf_eltwise(lambda a, b: a * b, attrs={'type': 'Eltwise', 'operation': 'mul'})),
    'Rsqrt': node_pb_arg(make_tf_eltwise(lambda v: np.reciprocal(np.sqrt(v)),
                                         attrs={'type': 'Power', 'power': -0.5, 'scale': 1, 'shift': 0})),
    'Neg': node_pb_arg(make_tf_eltwise(lambda v: -v, attrs={'type': 'Power', 'power': 1, 'scale': -1, 'shift': 0})),
    'Sub': node_pb_arg(make_tf_eltwise(lambda a, b: a - b)),
    'RealDiv': node_pb_arg(make_tf_eltwise(lambda a, b: a / b, attrs={'op': 'Div'})),
    'Relu': node_pb_arg(make_tf_eltwise(lambda v: np.maximum(0, v), attrs={'type': 'ReLU'})),  # 0 is an integer
    'RandomUniform': node_pb_arg(tf_random_uniform_ext),
    'Mean': node_pb_arg(tf_mean_ext),
    'BiasAdd': node_pb_arg(tf_bias_add_ext),
    'Reshape': node_pb_arg(tf_reshape_ext),
    'Squeeze': node_pb_arg(tf_squeeze_ext),
    'Shape': node_pb_arg(tf_shape_ext),
    'Softmax': node_pb_arg(tf_softmax_ext),
    'SpaceToBatchND': node_pb_arg(tf_space_to_batch_ext),
    'BatchToSpaceND': node_pb_arg(tf_batch_to_space_ext),
    'StopGradient': node_pb_arg(make_tf_eltwise(lambda v: v)),
    'Square': node_pb_arg(make_tf_eltwise(lambda a: a * a)),
    'Minimum': node_pb_arg(make_tf_eltwise(lambda a, b: np.minimum(a, b))),  # can use clamp if one argument is const
    'Maximum': node_pb_arg(make_tf_eltwise(lambda a, b: np.maximum(a, b), attrs={'type': 'Eltwise',
                                                                                 'operation': 'max'})),
    'Sum': node_pb_arg(tf_sum_ext),
    'Range': node_pb_arg(tf_range_ext),
    'ReadVariableOp': node_pb_arg(make_tf_eltwise(lambda v: v, attrs={'op': 'Identity'})),
    'PlaceholderWithDefault': node_pb_arg(make_tf_eltwise(lambda v: v, attrs={'op': 'Identity'}))
}


def common_tf_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.pb.name,
        'op': node.pb.op,
        'precision': 'FP32'  # TODO use real precision derived from the model
    }


def tf_op_extractor(node: Node, lowered_keys_map: dict):
    # all required attributes for the 'TFCustomSubgraphCall' are set during their initialization
    if (node.has('op') and node.op == 'TFCustomSubgraphCall') or (not node.has_valid('pb')):
        return True, node.graph.node[node.id]

    result = common_tf_fields(node)
    node.graph.node[node.id].update(result)
    supported = False
    op = result['op'].lower()
    if op in lowered_keys_map:
        op = lowered_keys_map[op]
        assert op in tf_op_extractors
        attrs = tf_op_extractors[op](node)
        if attrs:
            result.update(attrs)
            supported = True
    new_attrs = native_tf_node_extractor(node.pb)
    new_attrs.update(result)
    result = new_attrs
    return supported, result
