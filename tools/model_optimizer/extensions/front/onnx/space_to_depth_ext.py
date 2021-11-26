# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.space_to_depth import SpaceToDepth
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class SpaceToDepthFrontExtractor(FrontExtractorOp):
    op = 'SpaceToDepth'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        block_size = onnx_attr(node, 'blocksize', 'i', default=None)
        SpaceToDepth.update_node_stat(node, {'block_size': block_size})
        return cls.enabled
