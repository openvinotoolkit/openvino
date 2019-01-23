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

import numpy as np

from mo.front.caffe.extractors.reshape import reshape_ext
from mo.utils.unittest.extractors import FakeMultiParam


class FakeReshapeProtoLayer:
    def __init__(self, val):
        self.reshape_param = val


class Shape:
    def __init__(self, val):
        self.dim = val


class TestReshapeParsing(unittest.TestCase):
    def test_reshape_check_attrs(self):
        attrs = {
            'axis': 0,
            'num_axes': -1,
            'shape': Shape(np.array([0, -1])),
        }

        res = reshape_ext(FakeReshapeProtoLayer(FakeMultiParam(attrs)), None)
        exp_attrs = {
            'op': 'Reshape',
            'type': 'Reshape',
            'axis': 0,
            'num_axes': -1,
            'dim': [0, -1]
        }

        for key in exp_attrs.keys():
            if key == 'dim':
                np.testing.assert_equal(res[key], exp_attrs[key])
            else:
                self.assertEqual(res[key], exp_attrs[key])
