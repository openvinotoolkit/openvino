# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.space_to_depth import SpaceToDepth
from mo.front.extractor import FrontExtractorOp


class SpaceToDepthFrontExtractor(FrontExtractorOp):
    op = 'SpaceToDepth'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        block_size = node.pb.attr['block_size'].i
        data_format = node.pb.attr['data_format'].s.decode('utf-8')
        SpaceToDepth.update_node_stat(node, {'block_size': block_size, 'data_format': data_format})
        return cls.enabled
