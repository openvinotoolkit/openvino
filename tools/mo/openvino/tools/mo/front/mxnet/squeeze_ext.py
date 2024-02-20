# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.squeeze import Squeeze


class SqueezeExtractor(FrontExtractorOp):
    op = 'squeeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)

        Squeeze.update_node_stat(node, {'squeeze_dims': attrs.int("axis", None), 'keep_at_least_1d': True})
        return cls.enabled
