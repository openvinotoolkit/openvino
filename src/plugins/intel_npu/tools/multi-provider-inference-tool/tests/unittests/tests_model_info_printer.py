#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import json
import os
import shutil
import sys
import unittest

sys.path.append("../..")
from params import ModelInfo
from params import ModelInfoPrinter

class UtilsTests_model_info_printer_validation(unittest.TestCase):
    def setUp(self):
       self.sandbox_dir = "UtilsTests_model_info_printer_validation"
       self.model_info_string = '''{
    "input_0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [1,2,3,4]
    },
    "input_1": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [4,[1,40],2,1]
    },
    "input_2": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [4,"?",2,1]
    },
    "input_3": {
        "layout": ["N", "C", "H", "W"],
        "element_type": "float32",
        "shape": [4,-1,2,1]
    }
}'''

    def tearDown(self):
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def test_model_info_printer_validation(self):
        model_info = ModelInfo(self.model_info_string)
        model_printer = ModelInfoPrinter()
        printable_data = model_printer.serialize_model_info(self.sandbox_dir, "model.ext",
                                                            model_info)
        # validate model info consistency
        printable_data_json = json.loads(printable_data)
        for io_name in model_info.get_model_io_names():
            self.assertTrue(io_name in printable_data_json.keys())
            initial_io_info = model_info.get_model_io_info(io_name)
            for initial_io_key in initial_io_info.keys():
                self.assertTrue(initial_io_key in printable_data_json[io_name].keys())
                self.assertEqual(model_info.get_model_io_info(io_name)[initial_io_key], printable_data_json[io_name][initial_io_key])

    def test_model_info_printer_file_serialization_deserialization(self):
        model_info = ModelInfo(self.model_info_string)
        model_printer = ModelInfoPrinter()
        printable_data = model_printer.serialize_model_info(self.sandbox_dir,
                                                            os.path.join("a location", "of", "the","model.ext"),
                                                            model_info)

        serialized_model_file_path = None
        for file in os.listdir(self.sandbox_dir):
            self.assertTrue(file.endswith(".json"))
            serialized_model_file_path = os.path.join(self.sandbox_dir, file)

        self.assertTrue(os.path.isfile(serialized_model_file_path))
        deserialized_model_info = ModelInfo(serialized_model_file_path)
        self.assertAlmostEqual(deserialized_model_info.get_model_io_names(), model_info.get_model_io_names())
        for io_name in model_info.get_model_io_names():
            self.assertTrue(io_name in deserialized_model_info.get_model_io_names())
            initial_io_info = model_info.get_model_io_info(io_name)
            for initial_io_key in initial_io_info.keys():
                self.assertTrue(initial_io_key in deserialized_model_info.get_model_io_info(io_name).keys())
                self.assertEqual(model_info.get_model_io_info(io_name)[initial_io_key], deserialized_model_info.get_model_io_info(io_name)[initial_io_key])

if __name__ == '__main__':
    unittest.main()
