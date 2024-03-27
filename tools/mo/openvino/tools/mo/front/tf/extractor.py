# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.tf.extractors.concat import tf_concat_ext
from openvino.tools.mo.front.tf.extractors.fused_bn import tf_fused_bn_extractor
from openvino.tools.mo.front.tf.extractors.native_tf import native_tf_node_extractor
from openvino.tools.mo.front.tf.extractors.pack import tf_pack_ext
from openvino.tools.mo.front.tf.extractors.utils import get_tf_node_port
from openvino.tools.mo.graph.graph import Node


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
        edge_list.append(create_tf_edge(src_node_id, node.id, in_port))
    return edge_list


def create_tf_edge(src_node_id: str, dst_node_id: str, in_port: int):
    """
    Creates an edge for given nodes and input port.
    """
    src_node, src_port = get_tf_node_port(src_node_id)
    tensor_name = src_node + ":" + str(src_port)
    cf_flag = False
    if src_node[0] == '^':
        src_node = src_node[1:]
        cf_flag = True
    return (src_node, dst_node_id, {
        'in': in_port,
        'out': src_port,
        # debug anchor for a framework name, out port and tensor name
        'fw_tensor_debug_info': [(src_node_id, tensor_name)],
        'in_attrs': ['in', 'control_flow_edge', 'permutation'],
        'out_attrs': ['out', 'permutation'],
        'data_attrs': ['fw_tensor_debug_info'],
        'control_flow_edge': cf_flag
    })


def node_pb_arg(pb_extractor: callable):
    return lambda node: pb_extractor(node.pb)


tf_op_extractors = {
    'TFCustomSubgraphCall': node_pb_arg(lambda pb: None),
    'FusedBatchNorm': node_pb_arg(tf_fused_bn_extractor),
    'FusedBatchNormV2': node_pb_arg(tf_fused_bn_extractor),
    'FusedBatchNormV3': node_pb_arg(tf_fused_bn_extractor),
    'ConcatV2': node_pb_arg(tf_concat_ext),
    'Pack': node_pb_arg(tf_pack_ext),
}


def common_tf_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.pb.name,
        'op': node.pb.op,
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
