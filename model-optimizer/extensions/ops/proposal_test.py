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

import unittest

from extensions.ops.proposal import ProposalOp
from mo.front.common.extractors.utils import layout_attrs
from mo.front.common.partial_infer.utils import int64_array
from mo.graph.graph import Node
from mo.utils.unittest.graph import build_graph

nodes_attributes = {'proposal_input': {'kind': 'data', 'shape': None, 'value': None},
                    'proposal': {'type': 'Proposal', 'kind': 'op'},
                    'proposal_out_data_1': {'kind': 'data', 'shape': None, 'value': None},
                    'proposal_out_data_2': {'kind': 'data', 'shape': None, 'value': None},
                    'op_output': {'kind': 'op', 'op': 'Result'},
                    'op_output2': {'kind': 'op', 'op': 'Result'},
                    }


class TestProposal(unittest.TestCase):
    def test_proposal_infer_one_output(self):
        graph = build_graph(nodes_attributes,
                            [('proposal_input', 'proposal'),
                             ('proposal', 'proposal_out_data_1'),
                             ('proposal_out_data_1', 'op_output')
                             ],
                            {'proposal_input': {'shape': int64_array([1, 3, 227, 227])},
                             'proposal': {'post_nms_topn': 2, **layout_attrs()}
                             })

        proposal_node = Node(graph, 'proposal')
        ProposalOp.proposal_infer(proposal_node)

        self.assertListEqual([1 * 2, 5], list(graph.node['proposal_out_data_1']['shape']))

    def test_proposal_infer_two_outputs(self):
        graph = build_graph(nodes_attributes,
                            [('proposal_input', 'proposal'),
                             ('proposal', 'proposal_out_data_1'),
                             ('proposal', 'proposal_out_data_2'),
                             ('proposal_out_data_1', 'op_output'),
                             ('proposal_out_data_2', 'op_output')
                             ],
                            {'proposal_input': {'shape': int64_array([1, 3, 227, 227])},
                             'proposal': {'post_nms_topn': 2, **layout_attrs()}
                             })

        proposal_node = Node(graph, 'proposal')
        ProposalOp.proposal_infer(proposal_node)

        self.assertListEqual(list([1 * 2, 5]), list(graph.node['proposal_out_data_1']['shape']))
        self.assertListEqual(list([1 * 2]), list(graph.node['proposal_out_data_2']['shape']))
