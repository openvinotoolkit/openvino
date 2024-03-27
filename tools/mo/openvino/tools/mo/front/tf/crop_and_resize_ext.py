# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.roipooling import ROIPooling


class CropAndResizeFrontExtractor(FrontExtractorOp):
    op = 'CropAndResize'
    enabled = True

    @classmethod
    def extract(cls, node):
        # update the attributes of the node and force 'op' to be 'CropAndResize' so extension that merges two of its
        # inputs would be called
        method = node.pb.attr['method'].s.decode('utf-8')
        if method != 'bilinear':
            log.warning(
                'The crop and resize method "{}" for node "{}" is not supported.'.format(method, node.soft_get('name')))
            return False
        ROIPooling.update_node_stat(node, {'spatial_scale': 1, 'op': 'CropAndResize', 'method': method})
        return cls.enabled
