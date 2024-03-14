# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.pad import AttributedPad


class PadFrontExtractor(FrontExtractorOp):
    op = 'Pad'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        pads = mo_array(list(attrs.tuple('pad_width', int, None)))
        pads = pads.reshape([-1, 2])
        value = attrs.float('constant_value', 0.0)

        node_attrs = {
            'pads': pads,
            'mode': attrs.str('mode', None),
            'fill_value': value,
        }

        AttributedPad.update_node_stat(node, node_attrs)
        return cls.enabled
