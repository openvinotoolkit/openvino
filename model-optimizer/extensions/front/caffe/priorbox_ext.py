# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.priorbox import PriorBoxOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp


class PriorBoxFrontExtractor(FrontExtractorOp):
    op = 'PriorBox'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.prior_box_param

        variance = param.variance
        if len(variance) == 0:
            variance = [0.1]

        update_attrs = {
            'aspect_ratio': np.array(param.aspect_ratio),
            'min_size': np.array(param.min_size),
            'max_size': np.array(param.max_size),
            'flip': int(param.flip),
            'clip': int(param.clip),
            'variance': list(variance),
            'img_size': param.img_size,
            'img_h': param.img_h,
            'img_w': param.img_w,
            'step': param.step,
            'step_h': param.step_h,
            'step_w': param.step_w,
            'offset': param.offset,
        }

        # these params can be omitted in caffe.proto and in param as consequence,
        # so check if it is set or set to default
        fields = [field[0].name for field in param.ListFields()]
        if 'density' in fields:
            update_attrs['density'] = np.array(param.density)
        if 'fixed_size' in fields:
            update_attrs['fixed_size'] = np.array(param.fixed_size)
        if 'fixed_ratio' in fields:
            update_attrs['fixed_ratio'] = np.array(param.fixed_ratio)

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        PriorBoxOp.update_node_stat(node, mapping_rule)
        return cls.enabled
