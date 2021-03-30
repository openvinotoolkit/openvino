# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from extensions.ops.depth_to_space import DepthToSpaceOp
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class DepthToSpaceFrontExtractor(FrontExtractorOp):
    op = 'DepthToSpace'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node
        node_name = node.soft_get('name', node.id)
        block_size = onnx_attr(node, 'blocksize', 'i', default=None)
        assert block_size is not None, \
            'DepthToSpace should have "blocksize" attribute specified for node {}'.format(node_name)
        onnx_mode = onnx_attr(node, 'mode', 's', default=b'DCR').decode()
        assert onnx_mode in ['DCR', 'CRD'], 'Unrecognized mode provided for DepthToSpace node {}'.format(node_name)
        if onnx_mode == 'DCR':
            mode = 'blocks_first'
        else:
            mode = 'depth_first'

        DepthToSpaceOp.update_node_stat(node, {'block_size': block_size, 'mode': mode})
        return cls.enabled
