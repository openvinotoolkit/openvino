"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import networkx as nx

from extensions.ops.proposal import ProposalOp
from mo.front.caffe.extractor import register_caffe_python_extractor
from mo.graph.graph import Graph
from mo.ops.op import Op


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
