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

import numpy as np

from mo.front.common.partial_infer.split import tf_split_infer
from mo.front.tf.extractors.concat import tf_concat_ext
from mo.front.tf.extractors.const import tf_const_ext
from mo.front.tf.extractors.eltwise import make_tf_eltwise
from mo.front.tf.extractors.fused_bn import tf_fused_bn_extractor
from mo.front.tf.extractors.lrn import tf_lrn_ext
from mo.front.tf.extractors.matmul import tf_matmul_ext, tf_batchmatmul_ext
from mo.front.tf.extractors.native_tf import native_tf_node_extractor
from mo.front.tf.extractors.pack import tf_pack_ext
from mo.front.tf.extractors.random_uniform import tf_random_uniform_ext
from mo.front.tf.extractors.space_to_batch import tf_space_to_batch_ext, tf_batch_to_space_ext
from mo.front.tf.extractors.split import tf_split_ext
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
    'LRN': node_pb_arg(tf_lrn_ext),
    'Split': node_pb_arg(lambda pb: tf_split_ext(pb, tf_split_infer)),
    'FusedBatchNorm': node_pb_arg(tf_fused_bn_extractor),
    'ConcatV2': node_pb_arg(tf_concat_ext),
    'MatMul': node_pb_arg(tf_matmul_ext),
    'BatchMatMul': node_pb_arg(tf_batchmatmul_ext),
    'BatchMatMulV2': node_pb_arg(tf_batchmatmul_ext),
    'Pack': node_pb_arg(tf_pack_ext),
    'Unpack': node_pb_arg(tf_unpack_ext),
    'Const': node_pb_arg(tf_const_ext),
    'Identity': node_pb_arg(make_tf_eltwise(lambda v: v, attrs={'identity': True})),
    'RandomUniform': node_pb_arg(tf_random_uniform_ext),
    'SpaceToBatchND': node_pb_arg(tf_space_to_batch_ext),
    'BatchToSpaceND': node_pb_arg(tf_batch_to_space_ext),
    'ReadVariableOp': node_pb_arg(make_tf_eltwise(lambda v: v, attrs={'identity': True})),
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
