# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.size import Size
from mo.front.extractor import FrontExtractorOp


class SizeExtractor(FrontExtractorOp):
    op = 'Size'
    enabled = True

    @classmethod
    def extract(cls, node):
        Size.update_node_stat(node, {'output_type': np.int64})
        return cls.enabled
