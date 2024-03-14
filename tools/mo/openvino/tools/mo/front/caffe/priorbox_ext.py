# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.priorbox import PriorBoxOp
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


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
            'aspect_ratio': mo_array(param.aspect_ratio),
            'min_size': mo_array(param.min_size),
            'max_size': mo_array(param.max_size),
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
            update_attrs['density'] = mo_array(param.density)
        if 'fixed_size' in fields:
            update_attrs['fixed_size'] = mo_array(param.fixed_size)
        if 'fixed_ratio' in fields:
            update_attrs['fixed_ratio'] = mo_array(param.fixed_ratio)

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        PriorBoxOp.update_node_stat(node, mapping_rule)
        return cls.enabled
