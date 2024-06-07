# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.constant_fill import ConstantFill
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class ConstantFillFrontExtractor(FrontExtractorOp):
    op = 'ConstantFill'
    enabled = True

    @classmethod
    def extract(cls, node):

        value = onnx_attr(node, 'value', 'f', default=float(0.0))
        input_as_shape = onnx_attr(node, 'input_as_shape', 'i')
        extra_shape = onnx_attr(node, 'extra_shape', 'ints')
        shape = onnx_attr(node, 'shape', 'ints')
        dtype = onnx_attr(node, 'dtype', 'i', 1)

        assert input_as_shape
        assert extra_shape is None
        assert shape is None
        assert dtype == 1

        attrs = {
            'fill_value': value,
            'input_as_shape': input_as_shape,
        }

        ConstantFill.update_node_stat(node, attrs)
        return cls.enabled
