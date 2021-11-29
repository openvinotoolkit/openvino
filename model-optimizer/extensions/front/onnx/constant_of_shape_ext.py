# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from onnx import numpy_helper

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.constant_of_shape import ConstantOfShape


class ConstantOfShapeExtractor(FrontExtractorOp):
    op = 'ConstantOfShape'
    enabled = True

    @classmethod
    def extract(cls, node):
        fill_value = onnx_attr(node, 'value', 't', default=np.array([0.0]), dst_type=lambda x: numpy_helper.to_array(x))

        ConstantOfShape.update_node_stat(node, {'fill_value': fill_value})
        return cls.enabled
