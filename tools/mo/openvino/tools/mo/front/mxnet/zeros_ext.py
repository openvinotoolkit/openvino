# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.ops.const import Const


class ZerosFrontExtractor(FrontExtractorOp):
    op = '_zeros'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        shape = list(attrs.tuple('shape', int, None))
        dtype = attrs.tuple('dtype', str, None)
        if dtype and len(dtype) == 1:
            dtype = dtype[0]
        else:
            dtype = np.float32
        zero_shapes = []
        for i, s in enumerate(shape):
            if s == 0:
                shape[i] = 1
                zero_shapes.append(i)

        update_attrs = {
            'shape': np.ndarray(shape),
            'value': np.zeros(shape, dtype=dtype),
            'zero_shapes': zero_shapes
        }

        # update the attributes of the node
        Const.update_node_stat(node, update_attrs)
        return cls.enabled
