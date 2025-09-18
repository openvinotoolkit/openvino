#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import os
import sys
import unittest

sys.path.append("../..")
from common.provider_description import ModelInfo

class UtilsTests_model_info_validation(unittest.TestCase):

    def setUp(self):
        test_string = '''{
    "input_0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [1,2,3,4]
    },
    "input_1": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [4,3,2,1]
    },
    "input_2": {
        "layout": ["N","C","H","W"],
        "element_type": "float32",
        "shape": [6,5,2,3]
    }
}'''
        self.model_info = ModelInfo(test_string)

    def test_model_info_loading(self):
        model_name = "there/is/my_test_model"
        self.model_info.set_model_name(model_name)
        self.assertEqual(self.model_info.model_name, model_name)

        io_info = self.model_info.get_model_io_info("input_0")
        self.assertEqual(io_info["shape"], [1,2,3,4])
        self.assertEqual(io_info["layout"], "NCHW")
        io_info = self.model_info.get_model_io_info("input_1")
        self.assertEqual(io_info["shape"], [4,3,2,1])
        self.assertEqual(io_info["layout"], "NCHW")
        io_info = self.model_info.get_model_io_info("input_2")
        self.assertEqual(io_info["shape"], [6,5,2,3])
        self.assertEqual(io_info["layout"], "NCHW")
        with self.assertRaises(RuntimeError):
            self.model_info.get_model_io_info("my_nonexisting_input")

    def test_model_info_input_insertion(self):
        new_shape = [10,20,30,40]
        input_to_insert_name = "input_0"
        self.model_info.insert_info(input_to_insert_name, {"shape":new_shape})
        io_info = self.model_info.get_model_io_info(input_to_insert_name)
        self.assertEqual(io_info["shape"], new_shape)

    def test_model_info_input_data_update(self):
        input_to_update_name = "input_1"
        updated_input_param_name = "my_new_param_to_update"
        updated_input_param_value = "my_new_param_value"
        self.model_info.update_info(input_to_update_name, {updated_input_param_name: updated_input_param_value})
        io_info = self.model_info.get_model_io_info(input_to_update_name)

        self.assertTrue(updated_input_param_name in io_info.keys(), f"New updated param '{updated_input_param_name}' must appear among the input data: {io_info}")
        self.assertEqual(io_info[updated_input_param_name], updated_input_param_value)


class UtilsTests_dynamic_model_info_validation(unittest.TestCase):
    def setUp(self):
        test_string = '''{
    "input_0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [[1,66],2,3,4]
    },
    "input_1": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": ["?",3,2,1]
    },
    "input_2": {
        "layout": ["N","C","H","W"],
        "element_type": "float32",
        "shape": [-1,3,2,1]
    }
}'''
        self.model_info = ModelInfo(test_string)

    def test_model_info_dynamic_shape_loading(self):
        model_name = "there/is/my_test_model"
        self.model_info.set_model_name(model_name)
        self.assertEqual(self.model_info.model_name, model_name)

        io_info = self.model_info.get_model_io_info("input_0")
        self.assertEqual(io_info["shape"], [[1,66],2,3,4])
        self.assertEqual(io_info["layout"], "NCHW")
        io_info = self.model_info.get_model_io_info("input_1")
        self.assertEqual(io_info["shape"], ["?",3,2,1])
        self.assertEqual(io_info["layout"], "NCHW")
        io_info = self.model_info.get_model_io_info("input_2")
        self.assertEqual(io_info["shape"], [-1,3,2,1])
        self.assertEqual(io_info["layout"], "NCHW")
        with self.assertRaises(RuntimeError):
            self.model_info.get_model_io_info("my_nonexisting_input")

if __name__ == '__main__':
    unittest.main()
