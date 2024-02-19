# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from openvino.tools.mo.front.caffe.argmax_ext import ArgMaxFrontExtractor
from openvino.tools.mo.ops.argmax import ArgMaxOp, arg_ops_infer
from openvino.tools.mo.ops.op import Op
from unit_tests.utils.extractors import FakeMultiParam
from unit_tests.utils.graph import FakeNode


class FakeArgMaxProtoLayer:
    def __init__(self, val):
        self.argmax_param = val


class TestArgMaxExt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Op.registered_ops['ArgMax'] = ArgMaxOp

    def test_argmax_no_pb_no_ml(self):
        self.assertRaises(AttributeError, ArgMaxFrontExtractor.extract, None)

    @patch('openvino.tools.mo.front.caffe.argmax_ext.merge_attrs')
    def test_argmax_ext_ideal_numbers(self, merge_attrs_mock):
        params = {
            'out_max_val': True,
            'top_k': 100,
            'axis': 2
        }
        merge_attrs_mock.return_value = {
            **params
        }

        fake_pl = FakeArgMaxProtoLayer(FakeMultiParam(params))
        fake_node = FakeNode(fake_pl, None)

        ArgMaxFrontExtractor.extract(fake_node)

        exp_res = {
            'out_max_val': True,
            'top_k': 100,
            'axis': 2,
            'infer': arg_ops_infer,
            'remove_values_output': True,
        }

        for key in exp_res.keys():
            self.assertEqual(fake_node[key], exp_res[key])
