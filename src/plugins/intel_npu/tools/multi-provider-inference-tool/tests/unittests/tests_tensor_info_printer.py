#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import datetime
import json
import os
import shutil
import sys
import unittest

from pathlib import Path

sys.path.append("../..")

from utils import prepare_input_description
from common.converters import layout_to_str
from common.enums import InputSourceFileType
from common.provider_description import ModelInfo, TensorInfo, TensorsInfoPrinter
from common.source_description import FilesStorage


class UtilsTests_TIF_n_FS_integration(unittest.TestCase):

    def setUp(self):
        model_info_string = '''{
    "input_0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [1,2,3,4]
    },
    "input_1": {
        "layout": ["N","C","H","W"],
        "element_type": "float32",
        "shape": [4,3,2,1]
    },
    "output_0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [11,22,33,44]
    },
    "output_1": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [44,33,22,11]
    },
    "output_2": {
        "layout": "N",
        "element_type": "float32",
        "shape": [4]
    }
}'''
        inputs_description_string = '''{
    "input_0": {
        "files": ["something_for_input_0"],
        "type": "image",
        "convert": {
            "shape": "[1,2,3,4]",
            "element_type": "float32",
            "layout": ["N","C","H","W"]
        }
    },
    "input_1": {
        "files": ["something_for_input_1"],
        "type": "image",
        "convert": {
            "shape": "[10,20,30,40]",
            "element_type": "float32",
            "layout": "NCHW"
        }
    }
}'''
        self.model_info = ModelInfo(model_info_string)
        self.files_info = FilesStorage()
        self.files_info.parse_inputs(inputs_description_string)

        self.sandbox_dir = Path("UtilsTests_TIF_n_FS_integration")
        self.temporary_directories = []

    def tearDown(self):
        for d in self.temporary_directories:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def create_tmp_directory(self, directory_path):
        directory_path.mkdir(parents=True, exist_ok=True)
        self.temporary_directories.append(directory_path)

    @staticmethod
    def stub_get_tensor_info(model_name, model_io_info, input_file_info) -> TensorInfo:
        stub_io_description = prepare_input_description(input_file_info, model_io_info["shape"], model_io_info["element_type"], None)
        info = TensorInfo()
        info.info["bytes_size"] = 4
        info.info["data"] = bytearray([0x00, 0x01, 0x02, 0x03])
        info.info["element_type"] = stub_io_description["to_model_element_type"]
        info.info["shape"] = stub_io_description["to_shape"]
        info.info["model"] = model_name
        info.validate()
        return info

    @staticmethod
    def generate_input_tensor_info_for_printer(out_model_name, input_names, in_model_info, in_files_info):
        tensor_info = []
        for input_name in input_names:
            tensor_info_from_provider = UtilsTests_TIF_n_FS_integration.stub_get_tensor_info(
                        out_model_name,
                        in_model_info.get_model_io_info(input_name),
                        in_files_info.inputs()[input_name])
            tensor_info_from_provider.set_type("input")
            tensor_input_info = dict(in_model_info.get_model_io_info(input_name))
            tensor_input_info.update(tensor_info_from_provider.info)
            tensor_input_info["source"] = input_name
            tensor_input_info["input_files"] = in_files_info.inputs()[input_name]
            tensor_info.append(tensor_input_info)
        return tensor_info

    @staticmethod
    def generate_output_tensor_info_for_printer(out_model_name, output_names, in_model_info):
        tensor_info = []
        for output_name in output_names:
            tensor_info_from_provider = UtilsTests_TIF_n_FS_integration.stub_get_tensor_info(
                        out_model_name,
                        in_model_info.get_model_io_info(output_name),
                        {"type" : InputSourceFileType.bin.name})
            tensor_info_from_provider.set_type("output")
            if output_name in in_model_info.get_model_io_names():
                tensor_input_info = dict(in_model_info.get_model_io_info(output_name))
                tensor_input_info.update(tensor_info_from_provider.info)
            else:
                tensor_input_info = tensor_info_from_provider.info

            tensor_input_info["source"] = output_name
            tensor_info.append(tensor_input_info)
        return tensor_info

    def test_serialize_input_tensors(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        valuable_inputs = ["input_0", "input_1"]
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_input_tensor_info_for_printer(
                            model_name,
                            valuable_inputs,
                            self.model_info,
                            self.files_info
                        )
        serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "input")

        self.assertEqual(len(serialized_file_paths), 2, "Files produced must be equal to a number of inputs")
        for f in serialized_file_paths:
            self.assertTrue(f.is_file())
            match = False;
            for i in valuable_inputs:
                match = match or (str(f).find(i) != -1)
            self.assertTrue(match)
        self.assertTrue(input_info_path.is_file())
        self.assertTrue(input_info_dump_path.is_file())

    def test_serialize_output_tensors(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_output_tensor_info_for_printer(model_name, ["output_0", "output_1", "output_2"],self.model_info)
        serialized_file_paths, output_info_path, output_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "output")

        self.assertEqual(len(serialized_file_paths), 3, "Files produced must be equal to a number of outputs")
        for f in serialized_file_paths:
            self.assertTrue(f.is_file())
        self.assertTrue(output_info_path.is_file())
        self.assertTrue(output_info_dump_path.is_file())

    def test_deserialize_output_tensors_path_validation(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        sandbox_dir = Path(datetime.datetime.now().strftime("my_test_dir_%Y%m%d_%H%M%S"))
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        with self.assertRaises(RuntimeError):
            printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        self.create_tmp_directory(sandbox_dir)
        with self.assertRaises(RuntimeError):
            printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        sandbox_model_dir = sandbox_dir / model_name
        self.create_tmp_directory(sandbox_model_dir)
        with self.assertRaises(RuntimeError):
            printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        sandbox_model_sources_info_file_path = os.path.join(sandbox_model_dir, TensorsInfoPrinter.get_file_name_to_dump_model_source("output"))
        candidate_tensors_info = {"my_output": {
                                            "element_type": "value",
                                            "shape": "[2,3,4,5]",
                                            "files": ["file"]
                                  }}

        # inject non-typical layout as well
        candidate_tensors_info["my_output"]["layout"] = ["N","C","H","W"]
        with open(sandbox_model_sources_info_file_path, "w") as file:
            json.dump(candidate_tensors_info, file)

        deserialized_info = printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        # corrected layout which is expected after deserilization instead of non-typical one
        candidate_tensors_info["my_output"]["layout"] = layout_to_str(candidate_tensors_info["my_output"]["layout"])
        self.assertEqual(deserialized_info, candidate_tensors_info)

    def test_serialize_input_tensors_as_new_inputs_image_case(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_input_tensor_info_for_printer(model_name, ["input_0", "input_1"], self.model_info, self.files_info)
        serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "input")

        # read generated JSON files as a file input and ensure
        # that the new input is equal to original sources
        restored_files_info = FilesStorage()
        restored_files_info.parse_inputs(input_info_path)

        # important fields must be matched
        for input_name, input_data in self.files_info.inputs().items():
            for major_data_field_name in input_data.keys():
                self.assertTrue(major_data_field_name in restored_files_info.files_per_input_json[input_name].keys())
                self.assertEqual(restored_files_info.files_per_input_json[input_name][major_data_field_name],
                                 input_data[major_data_field_name])
            if "convert" in input_data.keys():
                self.assertTrue("convert" in restored_files_info.files_per_input_json[input_name].keys())
        self.assertEqual(restored_files_info.files_per_input_json, self.files_info.files_per_input_json)

    def test_serialize_input_tensors_as_new_inputs_bin_case(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_input_tensor_info_for_printer(model_name, ["input_0", "input_1"], self.model_info, self.files_info)
        serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "input")

        # read generated JSON files as a file input and ensure
        # that the new "bin" input is compatible with original "image" sources
        restored_files_info = FilesStorage()
        restored_files_info.parse_inputs(input_info_dump_path)

        # cross compare original input as "image" and generated "bin"
        for input_name, input_data in self.files_info.inputs().items():
            if "convert" in input_data.keys():
                for major_data_field_name in input_data["convert"].keys():
                    self.assertEqual(restored_files_info.files_per_input_json[input_name][major_data_field_name],
                                     input_data["convert"][major_data_field_name])


class UtilsTests_TIF_n_FS_io_canonization_integration(unittest.TestCase):

    def setUp(self):
        model_info_string = '''{
    "input_>:<0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [1,2,3,4]
    },
    "input_/|\\\\1": {
        "layout": ["N","C","H","W"],
        "element_type": "float32",
        "shape": [4,3,2,1]
    },
    "input_?*\\"2": {
        "layout": ["N","C","H","W"],
        "element_type": "float32",
        "shape": [4,3,2,1]
    },
    ">i<n:p|u\\\\t/_?*\\"3": {
        "layout": ["N","C","H","W"],
        "element_type": "float32",
        "shape": [4,3,2,1]
    },
    "output_0": {
        "layout": "NCHW",
        "element_type": "float32",
        "shape": [11,22,33,44]
    }
}'''

        inputs_description_string = '''{
    "input_>:<0": {
        "files": ["something_for_input_0"],
        "type": "image",
        "convert": {
            "shape": "[1,2,3,4]",
            "element_type": "float32",
            "layout": ["N","C","H","W"]
        }
    },
    "input_/|\\\\1": {
        "files": ["something_for_input_1"],
        "type": "image",
        "convert": {
            "shape": "[10,20,30,40]",
            "element_type": "float32",
            "layout": "NCHW"
        }
    },
    "input_?*\\"2": {
        "files": ["something_for_input_2"],
        "type": "image",
        "convert": {
            "shape": "[10,20,30,40]",
            "element_type": "float32",
            "layout": "NCHW"
        }
    },
    ">i<n:p|u\\\\t/_?*\\"3": {
        "files": ["something_for_input_3"],
        "type": "image",
        "convert": {
            "shape": "[10,20,30,40]",
            "element_type": "float32",
            "layout": "NCHW"
        }
    }
}'''
        self.model_info = ModelInfo(model_info_string)
        self.files_info = FilesStorage()
        self.files_info.parse_inputs(inputs_description_string)

        self.sandbox_dir = Path("UtilsTests_TIF_n_FS_io_canonization_integration")
        self.temporary_directories = []

    def tearDown(self):
        for d in self.temporary_directories:
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(self.sandbox_dir, ignore_errors=True)

    def create_tmp_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)
        self.temporary_directories.append(directory_path)

    def test_serialize_input_tensors(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        valuable_inputs = ["input_>:<0", "input_/|\\1", "input_?*\"2", ">i<n:p|u\\t/_?*\"3"]
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_input_tensor_info_for_printer(
                            model_name,
                            valuable_inputs,
                            self.model_info, self.files_info
                      )
        serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "input")

        self.assertEqual(len(serialized_file_paths), 4, "Files produced must be equal to a number of inputs")
        for f in serialized_file_paths:
            self.assertTrue(f.is_file())
            # raw model inputs must not be found among serialized paths
            match = False;
            for i in valuable_inputs:
                match = match or (str(f).find(i) != -1)
            self.assertFalse(match)

            # it must consist of canonized symbols
            match = False
            for i in valuable_inputs:
                match = match or (str(f).find(TensorsInfoPrinter.canonize_io_name(i)) != -1)
            self.assertTrue(match)
        self.assertTrue(input_info_path.is_file())
        self.assertTrue(input_info_dump_path.is_file())

        # tests canonization

    def test_deserialize_output_tensors_path_validation(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        sandbox_dir = Path(datetime.datetime.now().strftime("my_test_dir_%Y%m%d_%H%M%S"))
        shutil.rmtree(sandbox_dir, ignore_errors=True)
        with self.assertRaises(RuntimeError):
            printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        self.create_tmp_directory(sandbox_dir)
        with self.assertRaises(RuntimeError):
            printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        sandbox_model_dir = sandbox_dir / model_name
        self.create_tmp_directory(sandbox_model_dir)
        with self.assertRaises(RuntimeError):
            printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        sandbox_model_sources_info_file_path = os.path.join(sandbox_model_dir, TensorsInfoPrinter.get_file_name_to_dump_model_source("output"))
        candidate_tensors_info = {"my_output": {
                                            "element_type": "value",
                                            "shape": "[2,3,4,5]",
                                            "files": ["file"]
                                  }}
        # inject non-typical layout as well
        candidate_tensors_info["my_output"]["layout"] = ["N","C","H","W"]
        with open(sandbox_model_sources_info_file_path, "w") as file:
            json.dump(candidate_tensors_info, file)

        deserialized_info = printer.deserialize_output_tensor_descriptions(sandbox_dir, model_name)

        # corrected layout which is expected after deserilization instead of non-typical one
        candidate_tensors_info["my_output"]["layout"] = layout_to_str(candidate_tensors_info["my_output"]["layout"])
        self.assertEqual(deserialized_info, candidate_tensors_info)

    def test_serialize_input_tensors_as_new_inputs_image_case(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_input_tensor_info_for_printer(
                            model_name,
                            ["input_>:<0", "input_/|\\1", "input_?*\"2", ">i<n:p|u\\t/_?*\"3"],
                            self.model_info,
                            self.files_info
                      )
        serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "input")

        # read generated JSON files as a file input and ensure
        # that the new input is equal to original sources
        restored_files_info = FilesStorage()
        restored_files_info.parse_inputs(input_info_path)

        # important fields must be matched
        for input_name, input_data in self.files_info.inputs().items():
            self.assertTrue(input_name in restored_files_info.files_per_input_json.keys())
            for major_data_field_name in input_data.keys():
                self.assertTrue(major_data_field_name in restored_files_info.files_per_input_json[input_name].keys())
                self.assertEqual(restored_files_info.files_per_input_json[input_name][major_data_field_name],
                                 input_data[major_data_field_name])
            if "convert" in input_data.keys():
                self.assertTrue("convert" in restored_files_info.files_per_input_json[input_name].keys())
        self.assertEqual(restored_files_info.files_per_input_json, self.files_info.files_per_input_json)

    def test_serialize_input_tensors_as_new_inputs_bin_case(self):
        printer = TensorsInfoPrinter()

        model_name = "my_model"
        tensor_info = UtilsTests_TIF_n_FS_integration.generate_input_tensor_info_for_printer(
                            model_name,
                            ["input_>:<0", "input_/|\\1", "input_?*\"2", ">i<n:p|u\\t/_?*\"3"],
                            self.model_info,
                            self.files_info
                      )
        serialized_file_paths, input_info_path, input_info_dump_path = printer.serialize_tensors_by_type(
                            self.sandbox_dir, tensor_info, "input")

        # read generated JSON files as a file input and ensure
        # that the new "bin" input is compatible with original "image" sources
        restored_files_info = FilesStorage()
        restored_files_info.parse_inputs(input_info_dump_path)

        # cross compare original input as "image" and generated "bin"
        for input_name, input_data in self.files_info.inputs().items():
            self.assertTrue(input_name in restored_files_info.files_per_input_json.keys())
            if "convert" in input_data.keys():
                for major_data_field_name in input_data["convert"].keys():
                    self.assertEqual(restored_files_info.files_per_input_json[input_name][major_data_field_name],
                                     input_data["convert"][major_data_field_name])

if __name__ == '__main__':
    unittest.main()
