# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.space_to_depth import SpaceToDepth
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class SpaceToDepthFrontExtractor(FrontExtractorOp):
    op = 'SpaceToDepth'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        block_size = onnx_attr(node, 'blocksize', 'i', default=None)
        SpaceToDepth.update_node_stat(node, {'block_size': block_size})
        return cls.enabled
