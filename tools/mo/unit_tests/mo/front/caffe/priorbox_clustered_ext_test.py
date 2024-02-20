# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np

from openvino.tools.mo.front.caffe.priorbox_clustered_ext import PriorBoxClusteredFrontExtractor
from openvino.tools.mo.ops.priorbox_clustered import PriorBoxClusteredOp
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakePriorBoxClusteredProtoLayer:
    def __init__(self, val):
        self.prior_box_param = val


class TestPriorBoxClusteredExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PriorBoxClustered'] = PriorBoxClusteredOp

    def test_priorboxclustered_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PriorBoxClusteredFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.priorbox_clustered_ext.merge_attrs')
    def test_priorboxclustered_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'width': '30.0',
            'height': '60.0',
            'clip': False,
            'flip': True,
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6'
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakePriorBoxClusteredProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        PriorBoxClusteredFrontExtractor.extract(fake_node)

        exp_res = {
            'op': 'PriorBoxClustered',
            'type': 'PriorBoxClustered',
            'width': '30.0',
            'height': '60.0',
            'clip': 0,
            'flip': 1,
            'variance': np.array(['0.2', '0.3', '0.2', '0.3']),
            'img_size': '300',
            'img_h': '0',
            'img_w': '0',
            'step': '0,5',
            'step_h': '0',
            'step_w': '0',
            'offset': '0.6'
        }

        for key in exp_res.keys():
            if key in ['width', 'height', 'variance']:
                np.testing.assert_equal(fake_node[key], exp_res[key])
            else:
                self.assertEqual(fake_node[key], exp_res[key])
