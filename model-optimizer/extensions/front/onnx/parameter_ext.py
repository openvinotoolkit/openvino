# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from extensions.ops.parameter import Parameter
from mo.front.extractor import FrontExtractorOp


class PlaceholderFrontExtractor(FrontExtractorOp):
    op = 'Parameter'
    enabled = True

    @classmethod
    def extract(cls, node):
        t_type = node.pb.type.tensor_type
        attrs = {
            'shape': np.array([d.dim_value for d in t_type.shape.dim], dtype=np.int64),
            'data_type': TENSOR_TYPE_TO_NP_TYPE[t_type.elem_type]
        }
        Parameter.update_node_stat(node, attrs)
        return cls.enabled
