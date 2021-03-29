# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.normalize import NormalizeOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class NormalizeFrontExtractor(FrontExtractorOp):
    op = 'Normalize'
    enabled = True

    @classmethod
    def extract(cls, node):
        across_spatial = onnx_attr(node, 'across_spatial', 'i', default=0)
        channel_shared = onnx_attr(node, 'channel_shared', 'i', default=0)
        eps = onnx_attr(node, 'eps', 'f', default=0)
        
        attrs = {'across_spatial': bool(across_spatial),
                 'channel_shared': bool(channel_shared),
                 'eps': eps,
                 'layout': 'NCHW'}

        # update the attributes of the node
        NormalizeOp.update_node_stat(node, attrs)
        return cls.enabled
