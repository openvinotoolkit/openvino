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
from unittest.mock import patch

from extensions.front.caffe.proposal_ext import ProposalFrontExtractor
from extensions.ops.proposal import ProposalOp
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode, FakeAttr
from mo.ops.op import Op


class FakeProposalProtoLayer:
    def __init__(self, val):
        self.proposal_param = val


class TestProposalExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Proposal'] = ProposalOp

    def test_proposal_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ProposalFrontExtractor.extract, None)

    @patch('extensions.front.caffe.proposal_ext.merge_attrs')
    def test_proposal_ext_ideal_numbers(self, merge_attrs):
        params = {
            'feat_stride': 1,
            'base_size': 16,
            'min_size': 16,
            'ratio': 1,
            'scale': 2,
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7
        }
        merge_attrs.return_value = {
            **params
        }

        fake_pl = FakeProposalProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)
        fake_node.graph.graph['cmd_params'] = FakeAttr(generate_experimental_IR_V10=False)

        ProposalFrontExtractor.extract(fake_node)

        exp_res = {
            'type': "Proposal",
            'feat_stride': 1,
            'base_size': 16,
            'min_size': 16,
            'ratio': 1,
            'scale': 2,
            'pre_nms_topn': 6000,
            'post_nms_topn': 300,
            'nms_thresh': 0.7,
            'infer': ProposalOp.proposal_infer
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
