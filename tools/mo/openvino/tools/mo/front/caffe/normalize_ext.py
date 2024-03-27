# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.normalize import NormalizeOp
from openvino.tools.mo.front.caffe.collect_attributes import collect_attributes
from openvino.tools.mo.front.caffe.extractors.utils import weights_biases
from openvino.tools.mo.front.extractor import FrontExtractorOp


class NormalizeFrontExtractor(FrontExtractorOp):
    op = 'Normalize'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.norm_param

        attrs = collect_attributes(param, enable_flattening_nested_params=True)
        attrs.update(weights_biases(False, node.model_pb))
        # update the attributes of the node
        NormalizeOp.update_node_stat(node, attrs)
        return cls.enabled
