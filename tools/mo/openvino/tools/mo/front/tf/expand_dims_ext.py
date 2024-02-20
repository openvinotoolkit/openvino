# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.graph.graph import Node
from openvino.tools.mo.ops.unsqueeze import Unsqueeze


class ExpandDimsExtractor(FrontExtractorOp):
    """
    Due to historical reasons the ExpandDims operation in the Model Optimizer has one input with data and the attribute
    which specifies the dimension to expand. But in the TensorFlow the ExpandDims operation has 2 inputs where the
    second input specifies the dimensions to expand. In the Model Optimizer this operation corresponds to the Unsqueeze.
    """
    op = 'ExpandDims'
    enabled = True

    @classmethod
    def extract(cls, node: Node):
        Unsqueeze.update_node_stat(node, {})
        return cls.enabled
