# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.interpolate import Interpolate
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp


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
