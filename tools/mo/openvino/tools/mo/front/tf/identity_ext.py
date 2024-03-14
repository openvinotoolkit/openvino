# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.identity import Identity, IdentityN
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.tf.extractors.utils import tf_dtype_extractor
from openvino.tools.mo.graph.graph import Node


class IdentityFrontExtractor(FrontExtractorOp):
    op = 'Identity'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {
            'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
        })
        return cls.enabled


class IdentityNFrontExtractor(FrontExtractorOp):
    op = 'IdentityN'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        dtypes = [tf_dtype_extractor(t) for t in node.pb.attr["T"].list.type]
        IdentityN.update_node_stat(node, {
            'data_types': dtypes,
            'in_ports_count': len(dtypes),
            'out_ports_count': len(dtypes),
        })
        return cls.enabled


class ReadVariableOpFrontExtractor(FrontExtractorOp):
    op = 'ReadVariableOp'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {
            'data_type': tf_dtype_extractor(node.pb.attr["T"].type),
        })
        return cls.enabled


class StopGradientExtractor(FrontExtractorOp):
    op = 'StopGradient'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Identity.update_node_stat(node, {'op': 'StopGradient'})
        return cls.enabled
