# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.psroipooling import DeformablePSROIPoolingOp, PSROIPoolingOp
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class DeformablePSROIPoolingFrontExtractor(FrontExtractorOp):
    op = '_contrib_DeformablePSROIPooling'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        spatial_scale = attrs.float('spatial_scale', None)
        group_size = attrs.int('group_size', 0)
        no_trans = attrs.bool('no_trans', False)
        trans_std = attrs.float('trans_std', 0)
        output_dim = attrs.int('output_dim', 0)
        part_size = attrs.int('part_size', 0)
        sample_per_part = attrs.int('sample_per_part', 1)
        pooled_size = attrs.int('pooled_size', 0)

        data = {
            'spatial_scale': spatial_scale,
            'mode': 'bilinear_deformable',
            'group_size': group_size,
            'output_dim': output_dim,
            'trans_std': trans_std,
            'part_size': part_size,
            'spatial_bins_x': sample_per_part,
            'spatial_bins_y': sample_per_part,
            'pooled_width': pooled_size,
            'pooled_height': pooled_size,
        }

        DeformablePSROIPoolingOp.update_node_stat(node, data)
        return cls.enabled
