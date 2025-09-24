#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import json
import os
import sys
import unittest

sys.path.append("../..")
from common.provider_description import TensorInfo


class UtilsTests_tensor_info_validation(unittest.TestCase):

    def setUp(self):
        test_string = """{
        "data": "<something>",
        "bytes_size": 123456,
        "model": "<my_model>",
        "element_type":"float32",
        "shape": [1,2,3,4]
    }"""
        self.tensor_info = TensorInfo()
        self.tensor_info.info = json.loads(test_string)
        self.tensor_info.validate()

    def test_info_info_type(self):
        for ttype in TensorInfo.types:
            self.tensor_info.set_type(ttype)
            self.assertEqual(self.tensor_info.get_type(), ttype)

        with self.assertRaises(RuntimeError):
            self.tensor_info.set_type("my_nonexisting_type")

    def test_missing_necessary_attrs(self):
        for attr in ["data", "element_type", "model", "shape"]:
            tensor_info_copy = copy.deepcopy(self.tensor_info)
            del tensor_info_copy.info[attr]

            with self.assertRaises(RuntimeError):
                tensor_info_copy.validate()

    def test_shape_correction(self):
        self.tensor_info.info["shape"] = "[10,20,30,40,50]"
        self.tensor_info.validate()
        self.assertEqual(self.tensor_info.info["shape"], [10, 20, 30, 40, 50])

    def test_layout_correction(self):
        self.tensor_info.info["layout"] = ["N", "C", "H", "W"]
        self.tensor_info.validate()
        self.assertEqual(self.tensor_info.info["layout"], "NCHW")


if __name__ == "__main__":
    unittest.main()
