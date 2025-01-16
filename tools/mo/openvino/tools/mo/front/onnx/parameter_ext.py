# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

from openvino.tools.mo.ops.parameter import Parameter
from openvino.tools.mo.front.common.partial_infer.utils import shape_array, dynamic_dimension_value
from openvino.tools.mo.front.extractor import FrontExtractorOp


class PlaceholderFrontExtractor(FrontExtractorOp):
    op = 'Parameter'
    enabled = True

    @classmethod
    def extract(cls, node):
        t_type = node.pb.type.tensor_type
        attrs = {
            'shape': shape_array([d.dim_value if (not hasattr(d, 'dim_param') or d.dim_param == '') and d.dim_value != 0
                                  else dynamic_dimension_value for d in t_type.shape.dim]),
            'data_type': TENSOR_TYPE_TO_NP_TYPE[t_type.elem_type]
        }
        Parameter.update_node_stat(node, attrs)
        return cls.enabled
