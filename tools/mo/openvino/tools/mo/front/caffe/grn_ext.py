# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.grn import GRNOp
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class GRNFrontExtractor(FrontExtractorOp):
    op = 'GRN'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.grn_param

        update_attrs = {
            'bias': param.bias,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        # update the attributes of the node
        GRNOp.update_node_stat(node, mapping_rule)
        return cls.enabled
