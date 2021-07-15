# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from extensions.ops.random_uniform import AttributedRandomUniform
from mo.front.extractor import FrontExtractorOp
from mo.graph.graph import Node
from mo.front.onnx.extractors.utils import onnx_attr
from mo.front.common.partial_infer.utils import int64_array


class RangeFrontExtractor(FrontExtractorOp):
    op = 'RandomUniform'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        shape = onnx_attr(node, 'shape', 'ints', default=None, dst_type=int64_array)
        AttributedRandomUniform.update_node_stat(node, {'shape': shape})
        return cls.enabled
