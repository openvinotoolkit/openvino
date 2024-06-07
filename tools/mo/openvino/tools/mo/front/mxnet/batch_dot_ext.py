# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.mxnet.extractors.utils import get_mxnet_layer_attrs
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.MatMul import MatMul


class BatchDotExt(FrontExtractorOp):
    """
    MXNet operation which computes batch matrix multiplication of x and y similar to TensorFlow or ONNX MatMul operation.

    Attributes:
        transpose_a - if true then transpose the first input before multiplication
        transpose_b - if true then transpose the second input before multiplication
    """
    op = 'batch_dot'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        attrs = get_mxnet_layer_attrs(node.symbol_dict)
        transpose_a = attrs.bool('transpose_a', False)
        transpose_b = attrs.bool('transpose_b', False)
        forward_stype = attrs.str('forward_stype', None)

        if forward_stype is not None:
            log.error("Node {} has non default value {} of attribute forward_stype."
                      "Model Optimizer conversion assumes default value = None".format(node.soft_get('name', node.id),
                                                                                       forward_stype),
                      extra={'is_warning': True})

        MatMul.update_node_stat(node, {
            'transpose_a': transpose_a,
            'transpose_b': transpose_b
        })
        return cls.enabled
