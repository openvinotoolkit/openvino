# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from extensions.front.caffe.proposal_ext import ProposalFrontExtractor
from extensions.ops.proposal import ProposalOp
from mo.ops.op import Op
from mo.utils.unittest.extractors import FakeMultiParam
from mo.utils.unittest.graph import FakeNode, FakeAttr


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
