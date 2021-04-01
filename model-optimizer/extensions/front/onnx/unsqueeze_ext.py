# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.expand_dims import ExpandDims


class UnsqueezeFrontExtractor(FrontExtractorOp):
    """
    Convert Unsqueeze layer to ExpandDims because the ExpandDims layer has fixed attribute with dimensions to unsqueeze.
    """
    op = 'Unsqueeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = np.array(onnx_attr(node, 'axes', 'ints', default=[]), dtype=np.int64)

        ExpandDims.update_node_stat(node, {'expand_axis': axis})
        return cls.enabled
