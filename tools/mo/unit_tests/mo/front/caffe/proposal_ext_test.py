# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.proposal_ext import ProposalFrontExtractor
from openvino.tools.mo.ops.proposal import ProposalOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeProposalProtoLayer:
    def __init__(self, val):
        self.proposal_param = val


class TestProposalExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['Proposal'] = ProposalOp

    def test_proposal_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ProposalFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.proposal_ext.merge_attrs')
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
