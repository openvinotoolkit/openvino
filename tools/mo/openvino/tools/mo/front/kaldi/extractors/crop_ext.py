# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.crop import Crop


class CropFrontExtractor(FrontExtractorOp):
    op = 'Crop'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        mapping_rule = {
            'dim': pb['dim'],
            'offset': pb['offset'],
            'axis': pb['axis'],
            'layout': 'NCHW'
        }

        Crop.update_node_stat(node, attrs=mapping_rule)
        return cls.enabled
