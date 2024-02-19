# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.ops.flatten import FlattenONNX


class FlattenFrontExtractor(FrontExtractorOp):
    op = 'Flatten'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', 1)
        attrs = {
            'axis': axis
        }

        FlattenONNX.update_node_stat(node, attrs)
        return cls.enabled
