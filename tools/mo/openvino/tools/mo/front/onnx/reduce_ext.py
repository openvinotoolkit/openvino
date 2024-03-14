# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import ReduceL1, ReduceL2, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr
from openvino.tools.mo.graph.graph import Node


def update_reduce_node_attrs_with(node: Node, c: callable):
    axis = onnx_attr(node, 'axes', 'ints', default=None)
    if axis is not None:
        axis = int64_array(axis)
    keep_dims = onnx_attr(node, 'keepdims', 'i', default=True)
    c.update_node_stat(node, {'axis': axis, 'keep_dims': keep_dims})


class ReduceL1Extractor(FrontExtractorOp):
    op = 'ReduceL1'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceL1)
        return cls.enabled


class ReduceL2Extractor(FrontExtractorOp):
    op = 'ReduceL2'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceL2)
        return cls.enabled


class ReduceMaxFrontExtractor(FrontExtractorOp):
    op = 'ReduceMax'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceMax)
        return cls.enabled


class ReduceMeanFrontExtractor(FrontExtractorOp):
    op = 'ReduceMean'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceMean)
        return cls.enabled


class ReduceMinFrontExtractor(FrontExtractorOp):
    op = 'ReduceMin'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceMin)
        return cls.enabled


class ReduceProdFrontExtractor(FrontExtractorOp):
    op = 'ReduceProd'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceProd)
        return cls.enabled


class ReduceSumFrontExtractor(FrontExtractorOp):
    op = 'ReduceSum'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        update_reduce_node_attrs_with(node, ReduceSum)
        return cls.enabled
