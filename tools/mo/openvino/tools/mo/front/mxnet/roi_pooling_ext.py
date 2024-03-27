# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.roipooling import ROIPooling


class ROIPoolingFrontExtractor(FrontExtractorOp):
    op = 'ROIPooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        spatial_scale = attrs.float("spatial_scale", None)
        pooled_size = attrs.tuple("pooled_size", int, (0, 0))
        data = {
            'type': 'ROIPooling',
            'spatial_scale': spatial_scale,
            'pooled_w': pooled_size[1],
            'pooled_h': pooled_size[0]
        }

        data.update(layout_attrs())

        # update the attributes of the node
        ROIPooling.update_node_stat(node, data)
        return cls.enabled
