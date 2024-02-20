# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import numpy as np


class PB(dict):
    def update_node(self):
        pass
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
                self.assertAlmostEqual(val, self.res[key], 7,
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
