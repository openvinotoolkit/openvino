# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.graph.graph import Node


def node_pb_arg(pb_extractor: callable):
    return lambda node: pb_extractor(node.pb)


onnx_op_extractors = {}


def common_onnx_fields(node: Node):
    return {
        'kind': 'op',
        'name': node.id,
         # no reliable name for an onnx node, name can be empty, so we use that surrogate built as ID in the loader
        'op': node.op if node.has_valid('op') else node.pb.op_type,
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
