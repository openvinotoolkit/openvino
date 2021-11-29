# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.correlation import CorrelationOp
from mo.front.caffe.collect_attributes import merge_attrs
from mo.front.common.extractors.utils import layout_attrs
from mo.front.extractor import FrontExtractorOp


class CorrelationFrontExtractor(FrontExtractorOp):
    op = 'Correlation'
    enabled = True

    @classmethod
    def extract(cls, node):
        proto_layer = node.pb
        param = proto_layer.correlation_param

        corr_type = 'caffe.CorrelationParameter.MULTIPLY'
        if param.correlation_type == 1:
            corr_type = 'caffe.CorrelationParameter.SUBTRACT'

        update_attrs = {
            'pad': param.pad,
            'kernel_size': param.kernel_size,
            'max_displacement': param.max_displacement,
            'stride_1': param.stride_1,
            'stride_2': param.stride_2,
            'single_direction': param.single_direction,
            'do_abs': int(param.do_abs),
            'correlation_type': corr_type,
        }

        mapping_rule = merge_attrs(param, update_attrs)

        mapping_rule.update(layout_attrs())

        # update the attributes of the node
        CorrelationOp.update_node_stat(node, mapping_rule)
        return cls.enabled
