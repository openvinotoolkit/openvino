# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.split import VariadicSplit, Split, AttributedSplit
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node


class SplitVExtractor(FrontExtractorOp):
    op = 'SplitV'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        VariadicSplit.update_node_stat(node, {'out_ports_count': node.pb.attr['num_split'].i,
                                              'swap_axis_and_split_size_inputs': True})
        return cls.enabled


class UnpackExtractor(FrontExtractorOp):
    op = 'Unpack'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        pb = node.pb
        AttributedSplit.update_node_stat(node,
                                         {
                                             'axis': pb.attr['axis'].i,
                                             'num_splits': pb.attr['num'].i,
                                             'squeeze_axis': True,
                                         })
        return cls.enabled


class SplitExtractor(FrontExtractorOp):
    op = 'Split'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        pb = node.pb
        Split.update_node_stat(node, {
            'num_splits': pb.attr['num_split'].i,
            'input_port': 1,
        })
        return cls.enabled
