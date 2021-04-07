# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from extensions.ops.non_max_suppression import NonMaxSuppression
from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr


class NonMaxSuppressionExtractor(FrontExtractorOp):
    op = 'NonMaxSuppression'
    enabled = True

    @classmethod
    def extract(cls, node):
        encoding_map = {0: 'corner', 1: 'center'}
        center_point_box = onnx_attr(node, 'center_point_box', 'i', default=0)
        NonMaxSuppression.update_node_stat(node, {'sort_result_descending': 0,
                                                  'output_type': np.int64,
                                                  'box_encoding': encoding_map[center_point_box]})
        return cls.enabled
