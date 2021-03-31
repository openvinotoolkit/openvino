# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.depth_to_space import DepthToSpaceOp
from mo.front.extractor import FrontExtractorOp


class DepthToSpaceFrontExtractor(FrontExtractorOp):
    op = 'DepthToSpace'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        block_size = node.pb.attr['block_size'].i
        data_format = node.pb.attr['data_format'].s.decode('utf-8')
        DepthToSpaceOp.update_node_stat(node, {'block_size': block_size, 'data_format': data_format})
        return cls.enabled
