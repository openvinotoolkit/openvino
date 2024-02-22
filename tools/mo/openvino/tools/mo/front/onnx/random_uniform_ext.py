# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.random_uniform import AttributedRandomUniform
from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.onnx.extractors.utils import onnx_attr, get_onnx_datatype_as_numpy
from openvino.tools.mo.graph.graph import Node


class RandomUniformFrontExtractor(FrontExtractorOp):
    op = 'RandomUniform'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        shape = onnx_attr(node, 'shape', 'ints', default=None, dst_type=int64_array)
        out_type = get_onnx_datatype_as_numpy(onnx_attr(node, 'dtype', 'i', default=1))
        seed = onnx_attr(node, 'seed', 'f', default=0.0)
        min_val = onnx_attr(node, 'low', 'f', default=0.0)
        max_val = onnx_attr(node, 'high', 'f', default=1.0)
        AttributedRandomUniform.update_node_stat(node, {'shape': shape,
                                                        'output_type': out_type,
                                                        'seed': seed,
                                                        'min_val': out_type(min_val),
                                                        'max_val': out_type(max_val)})
        return cls.enabled
