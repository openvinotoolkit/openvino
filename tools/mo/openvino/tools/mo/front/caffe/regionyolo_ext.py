# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.regionyolo import RegionYoloOp
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class RegionYoloFrontExtractor(FrontExtractorOp):
    op = 'RegionYolo'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.region_yolo_param
        flatten_param = proto_layer.flatten_param
        axis = flatten_param.axis
        end_axis = flatten_param.end_axis
        coords = param.coords
        classes = param.classes
        num = param.num
        update_attrs = {
            'coords': coords,
            'classes': classes,
            'num': num,
            'do_softmax': int(param.do_softmax),
            'anchors': mo_array(param.anchors),
            'mask': mo_array(param.mask)
        }

        flatten_attrs = {
            'axis': axis,
            'end_axis': end_axis
        }

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(flatten_attrs)
        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        RegionYoloOp.update_node_stat(node, mapping_rule)
        return cls.enabled
