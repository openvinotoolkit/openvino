# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.psroipooling import PSROIPoolingOp
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class PSROIPoolingFrontExtractor(FrontExtractorOp):
    op = 'PSROIPooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.psroi_pooling_param

        update_attrs = {
            'spatial_scale': param.spatial_scale,
            'output_dim': param.output_dim,
            'group_size': param.group_size,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        PSROIPoolingOp.update_node_stat(node, mapping_rule)
        return cls.enabled
