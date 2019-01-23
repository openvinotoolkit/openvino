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

import numpy as np


class PB(dict):
    __getattr__ = dict.get


class BaseExtractorsTestingClass(unittest.TestCase):
    expected = None
    res = None
    call_args = None
    expected_call_args = None

    def setUp(self):
        if hasattr(self, 'patcher') and self.patcher:  # pylint: disable=no-member
            patcher = patch(self.patcher)  # pylint: disable=no-member
            self.addCleanup(patcher.stop)
            self.infer_mock = patcher.start()

    def compare(self):
        if hasattr(self, 'infer_mock'):
            self.assertTrue(self.infer_mock.called)
        for key, val in self.expected.items():
            if key == "infer":
                self.assertEqual(self.call_args, self.expected_call_args)
            if type(val) is np.ndarray:
                np.testing.assert_equal(val, self.res[key])
            elif type(val) is list:
                self.assertTrue(np.all([val == self.res[key]]))
            else:
                self.assertEqual(val, self.res[key],
                                 "{} attribute comparison failed! Expected {} but {} given.".format(key, val,
                                                                                                    self.res[key]))


class FakeParam:
    def __init__(self, param_key, param_val):
        setattr(self, param_key, param_val)


class FakeMultiParam:
    def __init__(self, dict_values):
        self.dict_values = dict_values
        for (key, value) in dict_values.items():
            # if type(value) != dict:
            setattr(self, key, value)
            # else:
            #     setattr(self, key, FakeMultiParam(value))


class FakeBlob:
    def __init__(self, param_key, param_val):
        setattr(self, param_key, param_val)


class FakeModelLayer:
    def __init__(self, blobs_val):
        self.blobs = [FakeBlob('data', val) for val in blobs_val]


class FakeValue:
    def __init__(self, val):
        self.shape = val