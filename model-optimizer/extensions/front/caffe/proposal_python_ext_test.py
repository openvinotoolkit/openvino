"""
 Copyright (C) 2018-2020 Intel Corporation

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
from unittest.mock import patch

from extensions.front.caffe.proposal_python_ext import ProposalPythonFrontExtractor
from extensions.ops.proposal import ProposalOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode, FakeAttr
from mo.ops.op import Op


class FakeProposalPythonProtoLayer:
    def __init__(self, val):
        self.python_param = val


class TestProposalPythonExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Proposal'] = ProposalOp

    def test_proposal_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ProposalPythonFrontExtractor.extract, None)

    def test_proposal_ext_ideal_numbers(self):
        params = {
            'param_str': "'feat_stride': 16"
        }
        fake_pl = FakeProposalPythonProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)
        fake_node.graph.graph['cmd_params'] = FakeAttr(generate_experimental_IR_V10=False)

        ProposalPythonFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Proposal",
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [0.5, 1, 2],
            'scale': [8, 16, 32],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7,
            'infer': ProposalOp.proposal_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])

    def test_proposal_ext_scales(self):
        params = {
            'param_str': "'feat_stride': 16, 'scales': [1,2,3], 'ratios':[5, 6,7]"
        }
        fake_pl = FakeProposalPythonProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)
        fake_node.graph.graph['cmd_params'] = FakeAttr(generate_experimental_IR_V10=False)

        ProposalPythonFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Proposal",
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [5, 6, 7],
            'scale': [1, 2, 3],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7,
            'infer': ProposalOp.proposal_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])

    def test_proposal_ext_scale(self):
        params = {
            'param_str': "'feat_stride': 16, 'scale': [1,2,3], 'ratio':[5, 6,7]"
        }
        fake_pl = FakeProposalPythonProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)
        fake_node.graph.graph['cmd_params'] = FakeAttr(generate_experimental_IR_V10=False)

        ProposalPythonFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Proposal",
            'feat_stride': 16,
            'base_size': 16,
            'min_size': 16,
            'ratio': [5, 6, 7],
            'scale': [1, 2, 3],
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7,
            'infer': ProposalOp.proposal_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
