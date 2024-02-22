# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.caffe.proposal_python_ext import ProposalPythonFrontExtractor
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


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
