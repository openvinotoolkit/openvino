#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import copy
import json
import os
import pathlib
import shutil
import sys
import unittest

sys.path.append("../..")

from array import array
from common.provider_description import TensorsInfoPrinter
from tools.multi_provider_blobs_comparison import multi_provider_result_comparator

def generate_posix_file_paths_from_file_names(root_dir, provider_name, file_names):
    return [pathlib.Path(os.path.join(root_dir, provider_name, f)).as_posix() for f in file_names]

def generate_two_outputs_from_file_path_list(file_path_list_1, file_path_list_2):
    outputs_dump_data_str = '''{
    "output_0": {
        "shape": [1, 1],
        "element_type": "float16",
        "type": "bin",
        "files": ["''' + "\",\"".join(file_path_list_1) + '''"]
        },
    "output_1": {
        "shape": [1, 2],
        "element_type": "float16",
        "type": "bin",
        "files": ["''' + "\",\"".join(file_path_list_2) + '''"]
        }
    }'''
    return json.loads(outputs_dump_data_str)


class UtilsTests_Tools_multi_provider_result_comparator(unittest.TestCase):

    def setUp(self):
        self.sandbox_dir = "UtilsTests_Tools_multi_provider_result_comparator"
        self.temporary_directories = []

        self.provider_A_output_0_file_path_list = generate_posix_file_paths_from_file_names(self.sandbox_dir, "provider_A", ["output_0_file_0.blob", "output_0_file_1.blob"])
        self.provider_A_output_1_file_path_list = generate_posix_file_paths_from_file_names(self.sandbox_dir, "provider_A", ["output_1_file_0.blob", "output_1_file_1.blob"])
        self.provider_A_outputs_dump_data_json = generate_two_outputs_from_file_path_list(self.provider_A_output_0_file_path_list, self.provider_A_output_1_file_path_list)

        self.provider_B_output_0_file_path_list = generate_posix_file_paths_from_file_names(self.sandbox_dir, "provider_B", ["output_0_file_0.blob", "output_0_file_1.blob"])
        self.provider_B_output_1_file_path_list = generate_posix_file_paths_from_file_names(self.sandbox_dir, "provider_B", ["output_1_file_0.blob", "output_1_file_1.blob"])
        self.provider_B_outputs_dump_data_json = generate_two_outputs_from_file_path_list(self.provider_B_output_0_file_path_list, self.provider_B_output_1_file_path_list)

        self.provider_C_output_0_file_path_list = generate_posix_file_paths_from_file_names(self.sandbox_dir, "provider_C", ["output_0_file_0.blob", "output_0_file_1.blob"])
        self.provider_C_output_1_file_path_list = generate_posix_file_paths_from_file_names(self.sandbox_dir, "provider_C", ["output_1_file_0.blob", "output_1_file_1.blob"])
        self.provider_C_outputs_dump_data_json = generate_two_outputs_from_file_path_list(self.provider_C_output_0_file_path_list, self.provider_C_output_1_file_path_list)

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

    def test_one_or_more_provider_data_absent(self):
        provider_A = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        provider_B = copy.deepcopy(self.provider_B_outputs_dump_data_json)
        provider_C = copy.deepcopy(self.provider_C_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        metadata_file_provider_A = self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, provider_A)
        metadata_file_provider_B = self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)
        metadata_file_provider_C = self.generate_output_tensor_description_file_from_json("provider_C", usecase_num, model_name, provider_C)
        provider_A_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        provider_B_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        provider_C_base_dir = os.path.join(self.sandbox_dir, "provider_C")

        # Remove all metadata files except provider_C
        os.remove(metadata_file_provider_A)
        os.remove(metadata_file_provider_B)

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        # Only provider_C must be there
        self.assertTrue("providers" in result.keys())
        self.assertEqual(result["providers"], [provider_C_base_dir])
        self.assertTrue("status" in result.keys())
        self.assertEqual(len(result["status"]), 2)
        self.assertTrue("data" in result.keys())
        self.assertEqual(len(result["data"]), 1)
        self.assertTrue(provider_C_base_dir in result["data"].keys())

        # Generate provider_B data again
        metadata_file_provider_B = self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        # Only provider_A must be absent
        self.assertTrue("providers" in result.keys())
        self.assertEqual(result["providers"], [provider_B_base_dir, provider_C_base_dir])
        self.assertTrue("status" in result.keys())
        self.assertEqual(len(result["status"]), 1)
        self.assertTrue("data" in result.keys())
        self.assertEqual(len(result["data"]), 2)
        self.assertTrue(provider_B_base_dir in result["data"].keys())
        self.assertTrue(provider_C_base_dir in result["data"].keys())

    def test_compare_no_output_binary_files(self):
        provider_A = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        provider_B = copy.deepcopy(self.provider_B_outputs_dump_data_json)
        provider_C = copy.deepcopy(self.provider_C_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, provider_A)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)
        self.generate_output_tensor_description_file_from_json("provider_C", usecase_num, model_name, provider_C)
        provider_A_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        provider_B_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        provider_C_base_dir = os.path.join(self.sandbox_dir, "provider_C")

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        for field in ["providers", "status", "data"]:
            self.assertTrue(field in result.keys())
        self.assertEqual(result["providers"], [provider_A_base_dir, provider_B_base_dir, provider_C_base_dir])
        self.assertEqual(len(result["status"]), 0)  # no errors on provider level
        self.assertEqual(len(result["data"]), 3)

        self.assertTrue(provider_A_base_dir in result["data"].keys())
        self.assertTrue(provider_B_base_dir in result["data"].keys())
        self.assertTrue(provider_C_base_dir in result["data"].keys())
        for field in ["outputs", "status", "data"]:
            self.assertTrue(field in result["data"][provider_A_base_dir].keys())
            self.assertTrue(field in result["data"][provider_B_base_dir].keys())
            self.assertTrue(field in result["data"][provider_C_base_dir].keys())
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_B_base_dir]["outputs"])
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_C_base_dir]["outputs"])

        files_per_output = {provider_A_base_dir: {"output_0": len(self.provider_A_output_0_file_path_list), "output_1": len(self.provider_A_output_1_file_path_list)},
                            provider_B_base_dir: {"output_0": len(self.provider_B_output_0_file_path_list), "output_1": len(self.provider_B_output_1_file_path_list)},
                            provider_C_base_dir: {"output_0": len(self.provider_C_output_0_file_path_list), "output_1": len(self.provider_C_output_1_file_path_list)}}

        for o_name in result["data"][provider_A_base_dir]["outputs"]:
            # each provider has the same set of output names
            self.assertTrue(o_name in result["data"][provider_A_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_B_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_C_base_dir]["data"].keys())

            # each output data must have important fields
            for field in ["status", "data", "files", "shape", "element_type"]:
                self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name].keys())

            # status must have two records regarding the two files for each output being missing
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["status"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["status"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["status"]), files_per_output[provider_C_base_dir][o_name])

            # no files in lists
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["files"]), 0)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["files"]), 0)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["files"]), 0)
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["data"]), 0)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["data"]), 0)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["data"]), 0)

    def test_compare_fail_not_enough_binary_files(self):
        provider_A = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        provider_B = copy.deepcopy(self.provider_B_outputs_dump_data_json)
        provider_C = copy.deepcopy(self.provider_C_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, provider_A)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)
        self.generate_output_tensor_description_file_from_json("provider_C", usecase_num, model_name, provider_C)
        provider_A_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        provider_B_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        provider_C_base_dir = os.path.join(self.sandbox_dir, "provider_C")

        # generate files for 1 output out of 2 outputs for each provider
        for f in [*self.provider_A_output_0_file_path_list, *self.provider_B_output_0_file_path_list, *self.provider_C_output_0_file_path_list]:
            with open(f,"w") as file:
                file.write("1234")

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        for field in ["providers", "status", "data"]:
            self.assertTrue(field in result.keys())
        self.assertEqual(result["providers"], [provider_A_base_dir, provider_B_base_dir, provider_C_base_dir])
        self.assertEqual(len(result["status"]), 0)  # no errors on provider level
        self.assertEqual(len(result["data"]), 3)

        self.assertTrue(provider_A_base_dir in result["data"].keys())
        self.assertTrue(provider_B_base_dir in result["data"].keys())
        self.assertTrue(provider_C_base_dir in result["data"].keys())
        for field in ["outputs", "status", "data"]:
            self.assertTrue(field in result["data"][provider_A_base_dir].keys())
            self.assertTrue(field in result["data"][provider_B_base_dir].keys())
            self.assertTrue(field in result["data"][provider_C_base_dir].keys())
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_B_base_dir]["outputs"])
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_C_base_dir]["outputs"])

        files_per_output = {provider_A_base_dir: {"output_0": len(self.provider_A_output_0_file_path_list), "output_1": len(self.provider_A_output_1_file_path_list)},
                            provider_B_base_dir: {"output_0": len(self.provider_B_output_0_file_path_list), "output_1": len(self.provider_B_output_1_file_path_list)},
                            provider_C_base_dir: {"output_0": len(self.provider_C_output_0_file_path_list), "output_1": len(self.provider_C_output_1_file_path_list)}}
        for o_name, status_message_count in zip(result["data"][provider_A_base_dir]["outputs"], [0, files_per_output[provider_A_base_dir]["output_0"]]):
            # each provider has the same set of output names
            self.assertTrue(o_name in result["data"][provider_A_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_B_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_C_base_dir]["data"].keys())

            # each output data must have important fields
            for field in ["status", "data", "files", "shape", "element_type"]:
                self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name].keys())

            # status must have no record for the first output and
            # two records regarding the two files for teh second output being missing
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["status"]), status_message_count)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["status"]), status_message_count)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["status"]), status_message_count)

            # no files in lists
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["files"]), files_per_output[provider_A_base_dir][o_name] - status_message_count)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["files"]), files_per_output[provider_B_base_dir][o_name] - status_message_count)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["files"]), files_per_output[provider_C_base_dir][o_name] - status_message_count)
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["data"]), files_per_output[provider_A_base_dir][o_name] - status_message_count)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["data"]), files_per_output[provider_B_base_dir][o_name] - status_message_count)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["data"]), files_per_output[provider_C_base_dir][o_name] - status_message_count)

    def test_compare_binary_files_correlation(self):
        provider_A = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        provider_B = copy.deepcopy(self.provider_B_outputs_dump_data_json)
        provider_C = copy.deepcopy(self.provider_C_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, provider_A)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)
        self.generate_output_tensor_description_file_from_json("provider_C", usecase_num, model_name, provider_C)
        provider_A_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        provider_B_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        provider_C_base_dir = os.path.join(self.sandbox_dir, "provider_C")

        for f in [*self.provider_A_output_0_file_path_list, *self.provider_B_output_0_file_path_list, *self.provider_B_output_1_file_path_list, *self.provider_C_output_0_file_path_list, *self.provider_C_output_1_file_path_list]:
            with open(f,"w") as file:
                file.write("1234")
        for f in self.provider_A_output_1_file_path_list:
            with open(f,"w") as file:
                file.write("4321")

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        for field in ["providers", "status", "data"]:
            self.assertTrue(field in result.keys())
        self.assertEqual(result["providers"], [provider_A_base_dir, provider_B_base_dir, provider_C_base_dir])
        self.assertEqual(len(result["status"]), 0)  # no errors on provider level
        self.assertEqual(len(result["data"]), 3)

        self.assertTrue(provider_A_base_dir in result["data"].keys())
        self.assertTrue(provider_B_base_dir in result["data"].keys())
        self.assertTrue(provider_C_base_dir in result["data"].keys())
        for field in ["outputs", "status", "data"]:
            self.assertTrue(field in result["data"][provider_A_base_dir].keys())
            self.assertTrue(field in result["data"][provider_B_base_dir].keys())
            self.assertTrue(field in result["data"][provider_C_base_dir].keys())
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_B_base_dir]["outputs"])
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_C_base_dir]["outputs"])

        files_per_output = {provider_A_base_dir: {"output_0": len(self.provider_A_output_0_file_path_list), "output_1": len(self.provider_A_output_1_file_path_list)},
                            provider_B_base_dir: {"output_0": len(self.provider_B_output_0_file_path_list), "output_1": len(self.provider_B_output_1_file_path_list)},
                            provider_C_base_dir: {"output_0": len(self.provider_C_output_0_file_path_list), "output_1": len(self.provider_C_output_1_file_path_list)}}
        for o_name, std_corr in zip(result["data"][provider_A_base_dir]["outputs"], [1, 0.7171800136566162]):
            # each provider has the same set of output names
            self.assertTrue(o_name in result["data"][provider_A_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_B_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_C_base_dir]["data"].keys())

            # each output data must have important fields
            for field in ["status", "data", "files", "shape", "element_type"]:
                self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name].keys())

            # status must not contain suspicious records
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["status"]), 0)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["status"]), 0)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["status"]), 0)

            # All files must have been found
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["files"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["files"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["files"]), files_per_output[provider_C_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["data"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["data"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["data"]), files_per_output[provider_C_base_dir][o_name])

            for f in result["data"][provider_A_base_dir]["data"][o_name]["files"]:
                self.assertTrue(f in result["data"][provider_B_base_dir]["data"][o_name]["data"].keys())
                self.assertTrue(f in result["data"][provider_C_base_dir]["data"][o_name]["data"].keys())

                for field in ["path", "std_correlation"]:
                    self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name]["data"][f].keys())
                    self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name]["data"][f].keys())
                    self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name]["data"][f].keys())

                self.assertEqual(result["data"][provider_A_base_dir]["data"][o_name]["data"][f]["std_correlation"], 1)
                self.assertAlmostEqual(result["data"][provider_B_base_dir]["data"][o_name]["data"][f]["std_correlation"], std_corr, places=4)
                self.assertAlmostEqual(result["data"][provider_C_base_dir]["data"][o_name]["data"][f]["std_correlation"], std_corr, places=4)

    def test_compare_binary_files_correlation_lack_of_files_provider_B(self):
        provider_A = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        provider_B = copy.deepcopy(self.provider_B_outputs_dump_data_json)
        provider_C = copy.deepcopy(self.provider_C_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, provider_A)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)
        self.generate_output_tensor_description_file_from_json("provider_C", usecase_num, model_name, provider_C)
        provider_A_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        provider_B_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        provider_C_base_dir = os.path.join(self.sandbox_dir, "provider_C")

        # Do not generate files for B, output 1
        for f in [*self.provider_A_output_0_file_path_list,
                  *self.provider_B_output_0_file_path_list,
                  *self.provider_C_output_0_file_path_list, *self.provider_C_output_1_file_path_list]:
            with open(f,"w") as file:
                file.write("1234")
        # A output 1 has unexpected std_corr
        for f in self.provider_A_output_1_file_path_list:
            with open(f,"w") as file:
                file.write("4321")

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        for field in ["providers", "status", "data"]:
            self.assertTrue(field in result.keys())
        self.assertEqual(result["providers"], [provider_A_base_dir, provider_B_base_dir, provider_C_base_dir])
        self.assertEqual(len(result["status"]), 0)  # no errors on provider level
        self.assertEqual(len(result["data"]), 3)

        self.assertTrue(provider_A_base_dir in result["data"].keys())
        self.assertTrue(provider_B_base_dir in result["data"].keys())
        self.assertTrue(provider_C_base_dir in result["data"].keys())
        for field in ["outputs", "status", "data"]:
            self.assertTrue(field in result["data"][provider_A_base_dir].keys())
            self.assertTrue(field in result["data"][provider_B_base_dir].keys())
            self.assertTrue(field in result["data"][provider_C_base_dir].keys())
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_B_base_dir]["outputs"])
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_C_base_dir]["outputs"])

        files_per_output = {provider_A_base_dir: {"output_0": len(self.provider_A_output_0_file_path_list), "output_1": len(self.provider_A_output_1_file_path_list)},
                            provider_B_base_dir: {"output_0": len(self.provider_B_output_0_file_path_list), "output_1": 0},
                            provider_C_base_dir: {"output_0": len(self.provider_C_output_0_file_path_list), "output_1": len(self.provider_C_output_1_file_path_list)}}
        provider_B_status_per_output = {"output_0": 0, "output_1":len(self.provider_B_output_1_file_path_list) }
        provider_B_file_found_per_output = {"output_0": True, "output_1":False}
        for o_name, std_corr in zip(result["data"][provider_A_base_dir]["outputs"], [1, 0.7171800136566162]):
            # each provider has the same set of output names
            self.assertTrue(o_name in result["data"][provider_A_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_B_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_C_base_dir]["data"].keys())

            # each output data must have important fields
            for field in ["status", "data", "files", "shape", "element_type"]:
                self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name].keys())

            # status must not contain records for provider A & C
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["status"]), 0)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["status"]), 0)
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["status"]), provider_B_status_per_output[o_name])

            # provider_B must report by setting no_found_data, which has owner provider_A
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["not_found_data"]), provider_B_status_per_output[o_name])
            for not_found_file, not_found_file_data in result["data"][provider_B_base_dir]["data"][o_name]["not_found_data"].items():
                self.assertTrue("provider" in not_found_file_data.keys())
                self.assertEqual(not_found_file_data["provider"], [provider_A_base_dir])

            # no files in lists
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["files"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["files"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["files"]), files_per_output[provider_C_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["data"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["data"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["data"]), files_per_output[provider_C_base_dir][o_name])

            for f in result["data"][provider_A_base_dir]["data"][o_name]["files"]:
                self.assertEqual(f in result["data"][provider_B_base_dir]["data"][o_name]["data"].keys(), provider_B_file_found_per_output[o_name])
                self.assertTrue(f in result["data"][provider_C_base_dir]["data"][o_name]["data"].keys())

                for field in ["path", "std_correlation"]:
                    self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name]["data"][f].keys())
                    if provider_B_file_found_per_output[o_name]:
                        self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name]["data"][f].keys())
                    self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name]["data"][f].keys())

                self.assertAlmostEqual(result["data"][provider_A_base_dir]["data"][o_name]["data"][f]["std_correlation"], 1, places=5)
                if provider_B_file_found_per_output[o_name]:
                    self.assertAlmostEqual(result["data"][provider_B_base_dir]["data"][o_name]["data"][f]["std_correlation"], std_corr, places=4)

                self.assertAlmostEqual(result["data"][provider_C_base_dir]["data"][o_name]["data"][f]["std_correlation"], std_corr, places=4)

    def test_compare_binary_files_correlation_lack_of_files_ref_provider_A(self):
        provider_A = copy.deepcopy(self.provider_A_outputs_dump_data_json)
        provider_B = copy.deepcopy(self.provider_B_outputs_dump_data_json)
        provider_C = copy.deepcopy(self.provider_C_outputs_dump_data_json)

        model_name = "model"
        usecase_num = 0
        self.generate_output_tensor_description_file_from_json("provider_A", usecase_num, model_name, provider_A)
        self.generate_output_tensor_description_file_from_json("provider_B", usecase_num, model_name, provider_B)
        self.generate_output_tensor_description_file_from_json("provider_C", usecase_num, model_name, provider_C)
        provider_A_base_dir = os.path.join(self.sandbox_dir, "provider_A")
        provider_B_base_dir = os.path.join(self.sandbox_dir, "provider_B")
        provider_C_base_dir = os.path.join(self.sandbox_dir, "provider_C")

        # Do not generate files for B, output 1
        for f in [*self.provider_A_output_0_file_path_list,
                  *self.provider_B_output_0_file_path_list,
                  *self.provider_C_output_0_file_path_list, *self.provider_C_output_1_file_path_list]:
            with open(f,"w") as file:
                file.write("1234")
        # A output 1 has unexpected std_corr
        for f in self.provider_B_output_1_file_path_list:
            with open(f,"w") as file:
                file.write("4321")

        result = multi_provider_result_comparator(provider_A_base_dir, [provider_B_base_dir, provider_C_base_dir], model_name, usecase_num)

        for field in ["providers", "status", "data"]:
            self.assertTrue(field in result.keys())
        self.assertEqual(result["providers"], [provider_A_base_dir, provider_B_base_dir, provider_C_base_dir])
        self.assertEqual(len(result["status"]), 0)  # no errors on provider level
        self.assertEqual(len(result["data"]), 3)

        self.assertTrue(provider_A_base_dir in result["data"].keys())
        self.assertTrue(provider_B_base_dir in result["data"].keys())
        self.assertTrue(provider_C_base_dir in result["data"].keys())
        for field in ["outputs", "status", "data"]:
            self.assertTrue(field in result["data"][provider_A_base_dir].keys())
            self.assertTrue(field in result["data"][provider_B_base_dir].keys())
            self.assertTrue(field in result["data"][provider_C_base_dir].keys())
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_B_base_dir]["outputs"])
        self.assertEqual(result["data"][provider_A_base_dir]["outputs"], result["data"][provider_C_base_dir]["outputs"])

        files_per_output = {provider_A_base_dir: {"output_0": len(self.provider_A_output_0_file_path_list), "output_1": 0},
                            provider_B_base_dir: {"output_0": len(self.provider_B_output_0_file_path_list), "output_1": len(self.provider_B_output_1_file_path_list)},
                            provider_C_base_dir: {"output_0": len(self.provider_C_output_0_file_path_list), "output_1": len(self.provider_C_output_1_file_path_list)}}
        provider_A_status_per_output = {"output_0": 0, "output_1":len(self.provider_A_output_1_file_path_list) }
        provider_A_file_found_per_output = {"output_0": True, "output_1":False}
        for o_name, std_corr in zip(result["data"][provider_A_base_dir]["outputs"], [1, 0.7171800136566162]):
            # each provider has the same set of output names
            self.assertTrue(o_name in result["data"][provider_A_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_B_base_dir]["data"].keys())
            self.assertTrue(o_name in result["data"][provider_C_base_dir]["data"].keys())

            # each output data must have important fields
            for field in ["status", "data", "files", "shape", "element_type"]:
                self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name].keys())
                self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name].keys())

            # status must not contain records for provider B & C
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["status"]), 0)
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["status"]), 0)
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["status"]), provider_A_status_per_output[o_name])

            # provider_B must report by setting no_found_data, which has owner provider_B, C
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["not_found_data"]), provider_A_status_per_output[o_name])
            for not_found_file, not_found_file_data in result["data"][provider_A_base_dir]["data"][o_name]["not_found_data"].items():
                self.assertTrue("provider" in not_found_file_data.keys())
                self.assertEqual(not_found_file_data["provider"], [provider_B_base_dir, provider_C_base_dir])

            # no files in lists
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["files"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["files"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["files"]), files_per_output[provider_C_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_A_base_dir]["data"][o_name]["data"]), files_per_output[provider_A_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_B_base_dir]["data"][o_name]["data"]), files_per_output[provider_B_base_dir][o_name])
            self.assertEqual(len(result["data"][provider_C_base_dir]["data"][o_name]["data"]), files_per_output[provider_C_base_dir][o_name])

            for f in result["data"][provider_A_base_dir]["data"][o_name]["files"]:
                self.assertTrue(f in result["data"][provider_B_base_dir]["data"][o_name]["data"].keys())
                self.assertTrue(f in result["data"][provider_C_base_dir]["data"][o_name]["data"].keys())

                for field in ["path", "std_correlation"]:
                    self.assertTrue(field in result["data"][provider_A_base_dir]["data"][o_name]["data"][f].keys())
                    self.assertTrue(field in result["data"][provider_B_base_dir]["data"][o_name]["data"][f].keys())
                    self.assertTrue(field in result["data"][provider_C_base_dir]["data"][o_name]["data"][f].keys())

                self.assertAlmostEqual(result["data"][provider_A_base_dir]["data"][o_name]["data"][f]["std_correlation"], 1, places=5)
                self.assertAlmostEqual(result["data"][provider_B_base_dir]["data"][o_name]["data"][f]["std_correlation"], std_corr, places=4)
                self.assertAlmostEqual(result["data"][provider_C_base_dir]["data"][o_name]["data"][f]["std_correlation"], std_corr, places=4)

if __name__ == '__main__':
    unittest.main()
