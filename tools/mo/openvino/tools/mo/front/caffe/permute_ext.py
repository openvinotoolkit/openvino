# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import mo_array
from openvino.tools.mo.ops.transpose import Transpose
from openvino.tools.mo.front.extractor import FrontExtractorOp


class PermuteFrontExtractor(FrontExtractorOp):
    op = 'permute'
    enabled = True

    @classmethod
    def extract(cls, node):
        order = node.pb.permute_param.order
        Transpose.update_node_stat(node, {'order': mo_array(order, dtype=np.int32)})
        return cls.enabled
