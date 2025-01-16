# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.MatMul import MatMul
from openvino.tools.mo.front.extractor import FrontExtractorOp


class BatchMatMulExtractor(FrontExtractorOp):
    op = 'BatchMatMul'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = node.pb.attr
        attrs = {
            'transpose_a': int(attr['adj_x'].b),
            'transpose_b': int(attr['adj_y'].b),
        }
        MatMul.update_node_stat(node, attrs)
        return cls.enabled


class BatchMatMulV2Extractor(FrontExtractorOp):
    op = 'BatchMatMulV2'
    enabled = True

    @classmethod
    def extract(cls, node):
        attr = node.pb.attr
        attrs = {
            'transpose_a': int(attr['adj_x'].b),
            'transpose_b': int(attr['adj_y'].b),
        }
        MatMul.update_node_stat(node, attrs)
        return cls.enabled
