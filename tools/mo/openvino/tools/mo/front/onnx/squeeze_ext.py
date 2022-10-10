# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.ops.squeeze import Squeeze


class SqueezeFrontExtractor(FrontExtractorOp):
    op = 'Squeeze'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = int64_array(onnx_attr(node, 'axes', 'ints', default=[]))
        # check if axes attribute was provided, which is required for onnx opset < 13.
        # Otherwise it means that onnx opset >= 13 and there are no attributes
        if axis.any():
            attrs = {
                'squeeze_dims': axis if len(axis) != 0 else None
            }
            # update the attributes of the node
            Squeeze.update_node_stat(node, attrs)
        else:
            Squeeze.update_node_stat(node)
        return cls.enabled
