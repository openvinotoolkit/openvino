# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.priorbox_clustered import PriorBoxClusteredOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp


class PriorBoxClusteredFrontExtractor(FrontExtractorOp):
    op = 'PriorBoxClustered'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.prior_box_param

        variance = param.variance
        if len(variance) == 0:
            variance = [0.1]

        update_attrs = {
            'width': list(param.width),
            'height': list(param.height),
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

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        PriorBoxClusteredOp.update_node_stat(node, mapping_rule)
        return cls.enabled
