# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.argmax import ArgMaxOp
from openvino.tools.mo.front.caffe.collect_attributes import merge_attrs
from openvino.tools.mo.front.extractor import FrontExtractorOp


class ArgMaxFrontExtractor(FrontExtractorOp):
    op = 'ArgMax'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.argmax_param

        update_attrs = {
            'out_max_val': int(param.out_max_val),
            'top_k': param.top_k,
            'axis': param.axis,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        ArgMaxOp.update_node_stat(node, mapping_rule)
        # ArgMax must be converted to TopK but without the output with values
        ArgMaxOp.update_node_stat(node, {'remove_values_output': True})
        return cls.enabled
