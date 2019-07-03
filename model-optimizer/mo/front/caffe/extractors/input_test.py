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

from mo.front.caffe.extractors.input import input_ext, global_input_ext
from mo.utils.unittest.extractors import FakeParam


class FakeProtoLayer:
    def __init__(self, shape):
        self.input_param = FakeParam('shape', shape)


class TestInput(unittest.TestCase):
    @patch('mo.front.caffe.extractors.input.single_output_infer')
    def test_input_ext(self, single_output_infer_mock):
        single_output_infer_mock.return_value = {}
        shape = [FakeParam('dim', 1)]
        res = input_ext(FakeProtoLayer(shape), None)
        exp_res = {
            'op': 'Placeholder',
            'shape': [1],
            'infer': None
        }
        for i in exp_res.keys():
            if i == 'infer':
                res['infer'](None)
                self.assertTrue(single_output_infer_mock.called)

    @patch('mo.front.caffe.extractors.input.single_output_infer')
    def test_global_input_ext(self, single_output_infer_mock):
        single_output_infer_mock.return_value = {}
        res = global_input_ext(None, None)
        exp_res = {
            'op': 'Placeholder',
            'type': 'input',
            'infer': None
        }
        for i in exp_res.keys():
            if i == 'infer':
                res['infer'](None)
                self.assertTrue(single_output_infer_mock.called)
