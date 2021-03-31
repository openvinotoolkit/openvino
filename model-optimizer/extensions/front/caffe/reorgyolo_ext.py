# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.reorgyolo import ReorgYoloOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp


class ReorgYoloFrontExtractor(FrontExtractorOp):
    op = 'ReorgYolo'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.reorg_yolo_param

        stride = param.stride
        update_attrs = {
            'stride': stride,
        }
        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        ReorgYoloOp.update_node_stat(node, mapping_rule)
        return cls.enabled
