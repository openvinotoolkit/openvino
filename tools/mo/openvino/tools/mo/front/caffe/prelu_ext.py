# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.prelu import PReLU
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.caffe.extractors.utils import weights_biases
from openvino.tools.mo.front.common.extractors.utils import layout_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class PreluFrontExtractor(FrontExtractorOp):
    op = 'PReLU'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        pb_model = node.model_pb
        param = proto_layer.prelu_param

        update_attrs = {
            'channel_shared': int(param.channel_shared)
        }

        variance_norm_caffe_map = {
            0: 'caffe.FillerParameter.FAN_IN',
            1: 'caffe.FillerParameter.FAN_OUT',
            2: 'caffe.FillerParameter.AVERAGE'
        }

        if hasattr(param, 'filler'):
            update_attrs.update({
                'filler_type': param.filler.type,
                'filler_value': int(param.filler.value),
                'min': int(param.filler.min),
                'max': int(param.filler.max),
                'mean': int(param.filler.mean),
                'std': int(param.filler.std),
                'sparse': param.filler.sparse,
                'variance_norm': variance_norm_caffe_map[param.filler.variance_norm]
            })

        mapping_rule = merge_attrs(param, update_attrs)
        mapping_rule.update(weights_biases(False, pb_model))
        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        PReLU.update_node_stat(node, mapping_rule)
        return cls.enabled
