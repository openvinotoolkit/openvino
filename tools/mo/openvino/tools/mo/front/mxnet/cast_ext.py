# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.Cast import Cast
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node


class CastExtractor(FrontExtractorOp):
    op = 'Cast'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        Cast.update_node_stat(node, {'dst_type': np.dtype(attrs.str('dtype', 'float32'))})
        return cls.enabled
