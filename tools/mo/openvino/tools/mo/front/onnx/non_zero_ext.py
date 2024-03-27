# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.ops.non_zero import NonZero
from openvino.tools.mo.front.extractor import FrontExtractorOp


class NonZeroExtractor(FrontExtractorOp):
    op = 'NonZero'
    enabled = True

    @classmethod
    def extract(cls, node):
        NonZero.update_node_stat(node, {'output_type': np.int64})
        return cls.enabled
