"""
 Copyright (c) 2018 Intel Corporation

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

from mo.front.caffe.extractors.eltwise import eltwise_ext
from mo.utils.unittest.extractors import FakeMultiParam


class FakeProtoLayer:
    def __init__(self, operation, coeff=[1]):
        self.eltwise_param = FakeMultiParam({'operation': operation,
                                             'coeff': coeff})


class TestEltwise(unittest.TestCase):
    @patch('mo.front.caffe.extractors.eltwise.eltwise_infer')
    def test_eltwise_op_mul(self, eltwise_infer_mock):
        eltwise_infer_mock.return_value = {}
        res = eltwise_ext(FakeProtoLayer(0), None)
        exp_res = {
            'op': 'Mul',
            'operation': 'mul',
            'infer': None
        }

        for i in exp_res.keys():
            if i == 'infer':
                res['infer'](None)
                args = eltwise_infer_mock.call_args
                actual_lambda = args[0][1]
                self.assertTrue(eltwise_infer_mock.called)
                self.assertEqual(actual_lambda(3, 5), 3 * 5)
            else:
                self.assertEqual(res[i], exp_res[i])

    @patch('mo.front.caffe.extractors.eltwise.eltwise_infer')
    def test_eltwise_op_add(self, eltwise_infer_mock):
        eltwise_infer_mock.return_value = {}
        res = eltwise_ext(FakeProtoLayer(1, coeff=[0.39]), None)
        exp_res = {
            'op': 'Add',
            'operation': 'sum',
            'coeff': '0.39',
            'infer': None
        }

        for i in exp_res.keys():
            if i == 'infer':
                res['infer'](None)
                args = eltwise_infer_mock.call_args
                actual_lambda = args[0][1]
                self.assertTrue(eltwise_infer_mock.called)
                self.assertEqual(actual_lambda(3, 5), 3 + 5)
            else:
                self.assertEqual(res[i], exp_res[i])

    @patch('mo.front.caffe.extractors.eltwise.eltwise_infer')
    def test_eltwise_op_max(self, eltwise_infer_mock):
        eltwise_infer_mock.return_value = {}
        res = eltwise_ext(FakeProtoLayer(2), None)
        exp_res = {
            'op': 'Max',
            'operation': 'max',
            'infer': None
        }

        for i in exp_res.keys():
            if i == 'infer':
                res['infer'](None)
                args = eltwise_infer_mock.call_args
                actual_lambda = args[0][1]
                self.assertTrue(eltwise_infer_mock.called)
                self.assertEqual(actual_lambda(3, 5), 5)
            else:
                self.assertEqual(res[i], exp_res[i])

    def test_eltwise_op_exeption(self):
        self.assertRaises(Exception, eltwise_ext, FakeProtoLayer(4), None)
