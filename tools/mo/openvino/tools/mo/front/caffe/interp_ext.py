# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.interpolate import Interpolate
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp


class InterpFrontExtractor(FrontExtractorOp):
    op = 'Interp'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.interp_param

        update_attrs = {
            'height': param.height,
            'width': param.width,
            'zoom_factor': param.zoom_factor,
            'shrink_factor': param.shrink_factor,
        }

        mapping_rule = merge_attrs(param, update_attrs)
        mapping_rule.update({'fw': 'caffe', 'mode': 'linear', 'axes': int64_array([2, 3]),
                             'pads_begin': param.pad_beg, 'pads_end': param.pad_end, 'align_corners': 1})
        Interpolate.update_node_stat(node, mapping_rule)
        return cls.enabled
