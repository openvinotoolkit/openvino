# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.crop import crop_infer
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.crop import Crop


class CropFrontExtractor(FrontExtractorOp):
    op = 'Crop'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.crop_param
        mapping_rule = {
            'axis': param.axis,
            'offset': param.offset,
            'dim': None,  # set in infer
            'infer': crop_infer
        }
        # update the attributes of the node
        Crop.update_node_stat(node, mapping_rule)
        return cls.enabled
