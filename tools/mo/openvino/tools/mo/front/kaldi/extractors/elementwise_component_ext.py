# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.extractor import FrontExtractorOp
from openvino.tools.mo.front.kaldi.utils import read_token_value
from openvino.tools.mo.ops.eltwise_ninputs_in_1 import EltwiseNin1


class ElementwiseProductComponentFrontExtractor(FrontExtractorOp):
    op = 'elementwiseproductcomponent'
    enabled = True

    @classmethod
    def extract(cls, node):
        pb = node.parameters

        indim = read_token_value(pb, b'<InputDim>')
        outdim = read_token_value(pb, b'<OutputDim>')
        num_inputs = indim / outdim

        attrs = {'num_inputs': int(num_inputs),
                 'operation': 'mul'}

        EltwiseNin1.update_node_stat(node, attrs)
        return cls.enabled
