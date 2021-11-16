# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.ops.reshape import Reshape

class ReshapeFrontExtractor(FrontExtractorOp):
    op = 'Reshape'
    enabled = True

    @classmethod
    def extract(cls, node):
        dim = onnx_attr(node, 'shape', 'ints', None)
        if dim is not None:
            dim = np.array(dim, dtype=np.int64)
            Reshape.update_node_stat(node, {'dim': dim})
        else:
            Reshape.update_node_stat(node)
        return cls.enabled
