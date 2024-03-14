# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.front.caffe.extractor import register_caffe_python_extractor
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.ops.op import Op


class ProposalPythonExampleOp(Op):
    op = 'Proposal'

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': __class__.op,
            'op': __class__.op,
            'post_nms_topn': 300,
            'infer': ProposalOp.proposal_infer
        }

        super().__init__(graph, mandatory_props, attrs)


register_caffe_python_extractor(ProposalPythonExampleOp, 'rpn.proposal_layer.ProposalLayer.example')
Op.excluded_classes.append(ProposalPythonExampleOp)
