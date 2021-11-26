# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.mvn import MVNOnnx
from mo.front.common.partial_infer.utils import int64_array
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class MeanVarianceNormalizationExtractor(FrontExtractorOp):
    op = 'MeanVarianceNormalization'
    enabled = True

    @classmethod
    def extract(cls, node):
        axes = onnx_attr(node, 'axes', 'ints',
                         default=int64_array([0, 2, 3]),
                         dst_type=lambda x: np.array(x, dtype=np.int64))

        attrs = {
            'eps': 1e-9,
            'normalize_variance': 1,
            'axes': axes,
            'eps_mode': 'outside_sqrt',
        }

        MVNOnnx.update_node_stat(node, attrs)
        return cls.enabled
