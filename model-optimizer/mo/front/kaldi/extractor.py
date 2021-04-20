# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.graph.graph import Node


def node_pb_arg(pb_extractor):
    return lambda node: pb_extractor(node.parameters)


kaldi_type_extractors = {}


def common_kaldi_fields(node: Node) -> dict:
    layer_type = node.op
    return {
        'kind': 'op',
        'name': node.id,
        'op': layer_type,
        # generic code relies on op; it should be overridden by specific op extractor
        'infer': None,
    }


def kaldi_extractor(node: Node, lowered_keys_map: dict) -> (bool, dict):
    result = common_kaldi_fields(node)
    node.graph.node[node.id].update(result)

    supported = False
    layer_type = result['op']
    if layer_type in lowered_keys_map:
        result.update(lowered_keys_map[layer_type](node))
        supported = True

    return supported, result
