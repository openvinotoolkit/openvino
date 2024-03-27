# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.ops.shape import Shape


class ShapeFrontExtractor(FrontExtractorOp):
    op = 'Shape'
    enabled = True

    @classmethod
    def extract(cls, node):
        Shape.update_node_stat(node, {'output_type': np.int64})
        return cls.enabled
