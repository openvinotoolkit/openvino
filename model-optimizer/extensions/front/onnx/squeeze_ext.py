# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.squeeze import Squeeze


class SqueezeFrontExtractor(FrontExtractorOp):
    op = 'Squeeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = np.array(onnx_attr(node, 'axes', 'ints', default=[]), dtype=np.int64)

        attrs = {
            'squeeze_dims': axis if len(axis) != 0 else None
        }

        # update the attributes of the node
        Squeeze.update_node_stat(node, attrs)
        return cls.enabled
