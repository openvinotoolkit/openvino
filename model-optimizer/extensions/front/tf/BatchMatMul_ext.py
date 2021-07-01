# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.MatMul import MatMul
from mo.front.extractor import FrontExtractorOp


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
