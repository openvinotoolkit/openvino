"""
 Copyright (C) 2018-2021 Intel Corporation

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

from mo.front.tf.extractors.native_tf import native_tf_node_extractor
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
