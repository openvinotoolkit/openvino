# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.extractor import FrontExtractorOp
from mo.front.onnx.extractors.utils import onnx_attr
from mo.ops.softmax import SoftmaxONNX
from mo.ops.log_softmax import LogSoftmaxONNX


class SoftmaxExtractor(FrontExtractorOp):
    op = 'Softmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=1)
        SoftmaxONNX.update_node_stat(node, {'axis': axis})
        return cls.enabled


class LogSoftmaxExtractor(FrontExtractorOp):
    op = 'LogSoftmax'
    enabled = True

    @classmethod
    def extract(cls, node):
        axis = onnx_attr(node, 'axis', 'i', default=1)
        LogSoftmaxONNX.update_node_stat(node, {'axis': axis})
        return cls.enabled
