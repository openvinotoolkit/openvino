# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.ops.shape import Shape


class ShapeFrontExtractor(FrontExtractorOp):
    op = 'Shape'
    enabled = True

    @classmethod
    def extract(cls, node):
        Shape.update_node_stat(node, {'output_type': np.int64})
        return cls.enabled
