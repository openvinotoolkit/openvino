# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.range import Range
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node


class ArangeExt(FrontExtractorOp):
    op = '_arange'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        Range.update_node_stat(node, {
            'start': attrs.int('start', 0),
            'stop': attrs.int('stop', 0),
            'repeat': attrs.int('repeat', 1),
            'step': attrs.float('step', 1),
            'dtype': np.dtype(attrs.str('dtype ', 'float32'))
        })
        return cls.enabled
