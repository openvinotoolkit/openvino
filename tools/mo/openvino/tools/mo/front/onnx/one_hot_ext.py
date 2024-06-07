# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.one_hot import OneHot
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr


class OneHotExtractor(FrontExtractorOp):
    op = 'OneHot'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=-1)
        OneHot.update_node_stat(node, {'axis': axis, 'split_values': True})
        return cls.enabled
