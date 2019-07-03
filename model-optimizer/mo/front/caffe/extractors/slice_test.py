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

import numpy as np

from mo.front.caffe.extractors.slice import slice_ext
from mo.front.common.partial_infer.slice import caffe_slice_infer
from mo.utils.unittest.extractors import FakeMultiParam


class FakeProtoLayer:
    def __init__(self, val):
        self.slice_param = val


class TestSlice(unittest.TestCase):
    @patch('mo.front.caffe.extractors.slice.merge_attrs')
    def test_slice_ext(self, merge_attrs_mock):
        params = {
            'type': 'Slice',
            'axis': 2,
            'slice_point': np.array([256]),
            'slice_dim': 3,
            'infer': caffe_slice_infer
        }
        merge_attrs_mock.return_value = {
            **params,
            'test': 54,
            'test2': 'test3'
        }
        res = slice_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'type': 'Slice',
            'axis': 2,
            'slice_point': np.array([256]),
            'infer': caffe_slice_infer
        }
        for i in exp_res:
            self.assertEqual(res[i], exp_res[i])

    @patch('mo.front.caffe.extractors.slice.merge_attrs')
    def test_slice_ext_slice_dim(self, merge_attrs_mock):
        params = {
            'type': 'Slice',
            'axis': 1,
            'slice_point': np.array([256]),
            'slice_dim': 3,
            'infer': caffe_slice_infer
        }
        merge_attrs_mock.return_value = {
            **params,
            'axis': 3
        }
        res = slice_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'type': 'Slice',
            'axis': 3,
            'slice_point': np.array([256]),
            'infer': caffe_slice_infer
        }
        for i in exp_res:
            self.assertEqual(res[i], exp_res[i])

    @patch('mo.front.caffe.extractors.slice.merge_attrs')
    def test_slice_ext_no_params(self, merge_attrs_mock):
        params = {
            'type': 'Slice',
            'axis': 1,
            'slice_dim': 1,
            'slice_point': [],
            'infer': caffe_slice_infer
        }
        merge_attrs_mock.return_value = {
            'type': 'Slice',
            'axis': 1,
            'infer': caffe_slice_infer
        }
        res = slice_ext(FakeProtoLayer(FakeMultiParam(params)), None)
        exp_res = {
            'type': 'Slice',
            'axis': 1,
            'slice_point': [],
            'infer': caffe_slice_infer
        }
        for i in exp_res:
            self.assertEqual(res[i], exp_res[i])
