# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.transpose import Transpose
from mo.front.extractor import FrontExtractorOp


class PermuteFrontExtractor(FrontExtractorOp):
    op = 'permute'
    enabled = True

    @classmethod
    def extract(cls, node):
        order = node.pb.permute_param.order
        Transpose.update_node_stat(node, {'order': np.array(order, dtype=np.int32)})
        return cls.enabled
