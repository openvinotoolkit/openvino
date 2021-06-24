# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.lrn import AttributedLRN


class LRNFrontExtractor(FrontExtractorOp):
    op = 'LRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'alpha': onnx_attr(node, 'alpha', 'f', 1e-4),
            'beta': onnx_attr(node, 'beta', 'f', 0.75),
            'bias': onnx_attr(node, 'bias', 'f', 1.0),
            'local_size': onnx_attr(node, 'size', 'i', None),
        }
        AttributedLRN.update_node_stat(node, attrs)
        return cls.enabled
