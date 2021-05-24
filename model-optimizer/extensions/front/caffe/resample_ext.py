# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.interpolate import Interpolate
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp


class ResampleFrontExtractor(FrontExtractorOp):
    op = 'Resample'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.resample_param
        types = [
            "",
            'nearest',
            'linear',
            'cubic',
            'area',
        ]
        resample_type = types[param.type]

        update_attrs = {
            'antialias': int(param.antialias),
            'height': param.height,
            'width': param.width,
            'type': resample_type,
            'factor': param.factor,
            'fw': 'caffe',
        }

        mapping_rule = merge_attrs(param, update_attrs)
        mapping_rule['mode'] = mapping_rule['type']
        mapping_rule['axes'] = int64_array([2, 3])
        mapping_rule.pop('type')
        Interpolate.update_node_stat(node, mapping_rule)
        return cls.enabled
