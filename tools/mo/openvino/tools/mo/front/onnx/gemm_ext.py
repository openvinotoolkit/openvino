# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.ops.MatMul import GemmONNX
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class GemmFrontExtractor(FrontExtractorOp):
    op = 'Gemm'
    enabled = True

    @classmethod
    def extract(cls, node):
        attrs = {
            'alpha': onnx_attr(node, 'alpha', 'f', 1),
            'beta': onnx_attr(node, 'beta', 'f', 1),
            'transpose_a': onnx_attr(node, 'transA', 'i', 0),
            'transpose_b': onnx_attr(node, 'transB', 'i', 0),
            'broadcast_c': onnx_attr(node, 'broadcast', 'i', 1),
            # TODO: there is no axis in onnx operators.md
            'axis': int64_array(onnx_attr(node, 'axis', 'i', default=0))
        }
        GemmONNX.update_node_stat(node, attrs)
        return cls.enabled
