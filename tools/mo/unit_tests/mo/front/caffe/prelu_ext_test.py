# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.prelu_ext import PreluFrontExtractor
from openvino.tools.mo.ops.prelu import PReLU
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakePReLUProtoLayer:
    def __init__(self, val):
        self.prelu_param = val


class TestPreluExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['PReLU'] = PReLU

    def test_prelu_no_pb_no_ml(self):
        self.assertRaises(AttributeError, PreluFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.prelu_ext.merge_attrs')
    def test_reogyolo_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'channel_shared': False
        }

        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakePReLUProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        PreluFrontExtractor.extract(fake_node)

        exp_res = {
            'type': 'PReLU',
            'op': 'PReLU',
            'channel_shared': 0,
            'infer': PReLU.infer,
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
