# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.adaptive_avg_pooling import AdaptiveAvgPooling
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class AdaptiveAvgPooling2DFrontExtractor(FrontExtractorOp):
    op = '_contrib_AdaptiveAvgPooling2D'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        output_size = attrs.tuple("output_size", int, None)
        if len(output_size) == 1:
            output_size = (output_size[0], output_size[0])

        data = {
            'op': 'Pooling',
            'output_size': output_size
        }
        AdaptiveAvgPooling.update_node_stat(node, data)
        return cls.enabled
