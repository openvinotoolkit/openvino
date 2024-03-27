# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.psroipooling import PSROIPoolingOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class PSROIPoolingFrontExtractor(FrontExtractorOp):
    op = '_contrib_PSROIPooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        spatial_scale = attrs.float("spatial_scale", None)
        pooled_size = attrs.int("pooled_size", None)
        output_dim = attrs.int("output_dim", None)
        group_size = attrs.int("group_size", 0)

        if group_size == 0:
            group_size = pooled_size

        data = {
            'spatial_scale': spatial_scale,
            'output_dim': output_dim,
            'group_size': group_size,
        }

        # update the attributes of the node
        PSROIPoolingOp.update_node_stat(node, data)
        return cls.enabled
