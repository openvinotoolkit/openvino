#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import copy
import json
import os
import pathlib
import shutil
import sys
import unittest

sys.path.append("../..")

from params import TensorsInfoPrinter
from tools.blob_standard_correlation import get_blobs_std_correlation

class UtilsTests_Tools_get_blobs_std_correlation(unittest.TestCase):

    def setUp(self):
        self.sandbox_dir = "UtilsTests_Tools_get_blobs_std_correlation"
        self.temporary_directories = []

        provider_A_output_0_file_list = ["provider_A_output_0_file_0.blob", "provider_A_output_0_file_1.blob"]
        self.provider_A_output_0_file_path_list = [pathlib.Path(os.path.join(self.sandbox_dir, f)).as_posix() for f in provider_A_output_0_file_list]
        provider_A_output_1_file_list = ["provider_A_output_1_file_0.blob", "provider_A_output_1_file_1.blob"]
        self.provider_A_output_1_file_path_list = [pathlib.Path(os.path.join(self.sandbox_dir, f)).as_posix() for f in provider_A_output_1_file_list]
        self.provider_A_outputs_dump_data_str = '''{
        "output_0": {
            "shape": [1, 1],
            "element_type": "float32",
            "type": "bin",
            "files": ["''' + "\",\"".join(self.provider_A_output_0_file_path_list) + '''"]
            },
        "output_1": {
            "shape": [1, 2],
            "element_type": "float32",
            "type": "bin",
            "files": ["''' + "\",\"".join(self.provider_A_output_1_file_path_list) + '''"]
            }
        }'''
        self.provider_A_outputs_dump_data_json = json.loads(self.provider_A_outputs_dump_data_str)

        provider_B_output_0_file_list = ["provider_B_output_0_file_0.blob", "provider_B_output_0_file_1.blob"]
        self.provider_B_output_0_file_path_list = [pathlib.Path(os.path.join(self.sandbox_dir, f)).as_posix() for f in provider_B_output_0_file_list]
        provider_B_output_1_file_list = ["provider_B_output_1_file_0.blob", "provider_B_output_1_file_1.blob"]
        self.provider_B_output_1_file_path_list = [pathlib.Path(os.path.join(self.sandbox_dir, f)).as_posix() for f in provider_B_output_1_file_list]
        self.provider_B_outputs_dump_data_str = '''{
        "output_0": {
            "shape": [1, 1],
            "element_type": "float32",
            "type": "bin",
            "files": ["''' + "\",\"".join(self.provider_B_output_0_file_path_list) + '''"]
            },
        "output_1": {
            "shape": [1, 2],
            "element_type": "float32",
            "type": "bin",
            "files": ["''' + "\",\"".join(self.provider_B_output_1_file_path_list) + '''"]
            }
        }'''
        self.provider_B_outputs_dump_data_json = json.loads(self.provider_B_outputs_dump_data_str)

    def tearDown(self):
        for d in self.temporary_directories:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def create_tmp_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)
        self.temporary_directories.append(directory_path)

    def generate_output_tensor_description_file_from_json(self, provider_name, usecase_num, model_name, json_object):
        ttype = "output"

        provider_dir = os.path.join(self.sandbox_dir, provider_name, str(usecase_num))
        main_model_dir = os.path.join(provider_dir, model_name)
        self.create_tmp_directory(main_model_dir)

        generated_file_path = os.path.join(main_model_dir, TensorsInfoPrinter.get_file_name_to_dump_model_source(ttype))
        with open(generated_file_path, "w") as file:
            json.dump(json_object, file)
        return generated_file_path

    def test_compare_fail_no_metadata_files(self):
        lhs = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        rhs = copy.deepcopy(self.provider_B_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        metadata_file_provider_A = self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, lhs)
        metadata_file_provider_B = self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, rhs)
        lhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        rhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        os.remove(metadata_file_provider_A)
        os.remove(metadata_file_provider_B)

        printer = TensorsInfoPrinter()
        result = get_blobs_std_correlation(lhs_provider_base_dir, rhs_provider_base_dir, model_name, usecase_num)
        self.assertTrue("error_code" in result.keys())
        self.assertNotEqual(result["error_code"], 0)
        self.assertTrue("error_description" in result.keys())
        self.assertNotEqual(result["error_description"], "")

    def test_compare_fail_no_binary_files(self):
        lhs = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        rhs = copy.deepcopy(self.provider_B_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, lhs)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, rhs)
        lhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        rhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_B")

        printer = TensorsInfoPrinter()
        result = get_blobs_std_correlation(lhs_provider_base_dir, rhs_provider_base_dir, model_name, usecase_num)
        self.assertTrue("error_code" in result.keys())
        self.assertNotEqual(result["error_code"], 0)
        self.assertTrue("error_description" in result.keys())
        self.assertNotEqual(result["error_description"], "")

        self.assertTrue("lhs_files" in result.keys())
        self.assertTrue("rhs_files" in result.keys())
        self.assertTrue("std_correlation" in result.keys())
        # arrays of files must not be empty, as we filed these arrays before we check file existence.
        self.assertEqual(len(result["lhs_files"]), 1)
        self.assertEqual(len(result["rhs_files"]), 1)
        self.assertEqual(len(result["std_correlation"]), 1)
        self.assertEqual(result["std_correlation"][0], "NaN")

    def test_compare_fail_not_enough_binary_files(self):
        lhs = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        rhs = copy.deepcopy(self.provider_B_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, lhs)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, rhs)
        lhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        rhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_B")

        for f in self.provider_A_output_0_file_path_list:
            with open(f,"w") as file:
                file.write("1234")
        for f in self.provider_B_output_0_file_path_list:
            with open(f,"w") as file:
                file.write("1234")

        printer = TensorsInfoPrinter()
        result = get_blobs_std_correlation(lhs_provider_base_dir, rhs_provider_base_dir, model_name, usecase_num)
        self.assertTrue("error_code" in result.keys())
        self.assertNotEqual(result["error_code"], 0)
        self.assertTrue("error_description" in result.keys())
        self.assertNotEqual(result["error_description"], "")

        self.assertTrue("lhs_files" in result.keys())
        self.assertTrue("rhs_files" in result.keys())
        self.assertTrue("std_correlation" in result.keys())
        self.assertEqual(len(result["lhs_files"]), len(self.provider_A_output_0_file_path_list) + 1)
        self.assertEqual(len(result["rhs_files"]), len(self.provider_B_output_0_file_path_list) + 1)
        self.assertEqual(len(result["std_correlation"]), len(self.provider_A_output_0_file_path_list) + 1)
        self.assertEqual(result["std_correlation"][-1], "NaN")

    def test_compare_binary_files_correlation(self):
        lhs = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        rhs = copy.deepcopy(self.provider_B_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, lhs)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, rhs)
        lhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        rhs_provider_base_dir = os.path.join(self.sandbox_dir, "provider_B")

        for f in self.provider_A_output_0_file_path_list:
            with open(f,"w") as file:
                file.write("1234")
        for f in self.provider_B_output_0_file_path_list:
            with open(f,"w") as file:
                file.write("1234")
        for f in self.provider_A_output_1_file_path_list:
            with open(f,"w") as file:
                file.write("4321")
        for f in self.provider_B_output_1_file_path_list:
            with open(f,"w") as file:
                file.write("1234")
        printer = TensorsInfoPrinter()
        result = get_blobs_std_correlation(lhs_provider_base_dir, rhs_provider_base_dir, model_name, usecase_num)
        self.assertTrue("error_code" in result.keys())
        self.assertEqual(result["error_code"], 0)
        self.assertTrue("error_description" in result.keys())

        self.assertTrue("lhs_files" in result.keys())
        self.assertTrue("rhs_files" in result.keys())
        self.assertTrue("std_correlation" in result.keys())
        self.assertEqual(len(result["lhs_files"]), len(self.provider_A_output_0_file_path_list) + len(self.provider_A_output_1_file_path_list))
        self.assertEqual(len(result["rhs_files"]), len(self.provider_B_output_0_file_path_list) + len(self.provider_B_output_1_file_path_list))
        self.assertEqual(len(result["std_correlation"]), len(self.provider_A_output_0_file_path_list) + len(self.provider_B_output_1_file_path_list))
        self.assertEqual(result["std_correlation"][0:len(self.provider_A_output_0_file_path_list)], [1 for _ in range(0,len(self.provider_A_output_0_file_path_list))])
        self.assertNotEqual(result["std_correlation"][len(self.provider_A_output_0_file_path_list):], [1 for _ in range(0,len(self.provider_A_output_1_file_path_list))])

if __name__ == '__main__':
    unittest.main()
