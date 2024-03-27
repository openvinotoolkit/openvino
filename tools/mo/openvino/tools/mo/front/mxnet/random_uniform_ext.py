# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.random_uniform import AttributedRandomUniform
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs


class RandomUniformExtractor(FrontExtractorOp):
    op = '_random_uniform'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        shape = list(attrs.tuple("shape", int, None))
        high = attrs.float("high", 1.0)
        low = attrs.float("low", 0.0)
        out_type = attrs.dtype("dtype", np.float32)
        new_attrs = {'shape': shape, 'min_val': out_type(low), 'max_val': out_type(high), 'output_type': out_type}
        AttributedRandomUniform.update_node_stat(node, new_attrs)
        return cls.enabled
