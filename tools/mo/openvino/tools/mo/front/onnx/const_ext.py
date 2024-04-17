# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from onnx import numpy_helper
from onnx.numpy_helper import to_array

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.ops.const import Const


class ConstExtractor(FrontExtractorOp):
    op = 'Const'
    enabled = True

    @classmethod
    def extract(cls, node):
        value = to_array(node.pb_init)
        attrs = {
            'data_type': value.dtype,
            'value': value
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled


class ConstantExtractor(FrontExtractorOp):
    op = 'Constant'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb_value = onnx_attr(node, 'value', 't')
        value = numpy_helper.to_array(pb_value)

        attrs = {
            'data_type': value.dtype,
            'value': value,
        }
        Const.update_node_stat(node, attrs)
        return cls.enabled
