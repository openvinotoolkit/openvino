# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.ReduceOps import ReduceProd, ReduceAnd, ReduceMax, ReduceMean, ReduceSum, ReduceL2, \
    ReduceMin, ReduceLogicalOr
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node


class AllFrontExtractor(FrontExtractorOp):
    op = 'All'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        keep_dims = node.pb.attr['keep_dims'].b
        ReduceAnd.update_node_stat(node, {'keep_dims': keep_dims})
        return cls.enabled


class AnyExtractor(FrontExtractorOp):
    op = 'Any'
    enabled = True

    @classmethod
    def extract(cls, node):
        ReduceLogicalOr.update_node_stat(node, {'keep_dims':  node.pb.attr['keep_dims'].b})
        return cls.enabled


class MaxFrontExtractor(FrontExtractorOp):
    op = 'Max'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceMax.update_node_stat(node, {'keep_dims': node.pb.attr['keep_dims'].b})
        return cls.enabled


class MinFrontExtractor(FrontExtractorOp):
    op = 'Min'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceMin.update_node_stat(node, {'keep_dims': node.pb.attr['keep_dims'].b})
        return cls.enabled


class MeanExtractor(FrontExtractorOp):
    op = 'Mean'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceMean.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled


class ProdFrontExtractor(FrontExtractorOp):
    op = 'Prod'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceProd.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled


class SumFrontExtractor(FrontExtractorOp):
    op = 'Sum'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceSum.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled


class EuclideanNormFrontExtractor(FrontExtractorOp):
    op = 'EuclideanNorm'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        ReduceL2.update_node_stat(node, {'keep_dims': node.pb.attr["keep_dims"].b})
        return cls.enabled
