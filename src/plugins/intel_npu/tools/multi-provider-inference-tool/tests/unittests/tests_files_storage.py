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
from common.source_description import FilesStorage


class UtilsTests_file_input_validation(unittest.TestCase):
    def get_result_test_string_input_files_correct(test_string):
        s = FilesStorage()
        s.parse_inputs(test_string)
        return s.inputs()

    def get_result_test_string_input_files_without_important_fields(test_strings):
        s = FilesStorage()
        throw = False
        for incorrect_item, incorrect_input in test_strings.items():
            throw = False
            try:
                s.parse_inputs(incorrect_input)
            except RuntimeError as e:
                if str(e).find(incorrect_item) == -1:
                    return False, f"An exception description has no any mentions about '{incorrect_item}', but it has to"
                throw = True
            if not throw:
                return False, f"An exception complaining regarding '{incorrect_item}' must have been thrown"
        return True, ""

    # Parameteric tests
    test_string_input_files_correct_image_with_convert_test_string = (
        '{"image": {"files": ["something"],"type":"image","convert":{"shape":"[1,3,372,500]", "element_type":"float32"}}}'
    )

    def test_string_input_files_correct_image_with_convert(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["convert"]["shape"], [1, 3, 372, 500])

    test_string_input_files_correct_image_with_convert_good_shape_test_string = (
        '{"image": {"files": ["something"],"type":"image","convert":{"shape":[1,3,372,500], "element_type":"float32"}}}'
    )

    def test_string_input_files_correct_image_with_convert(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_good_shape_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["convert"]["shape"], [1, 3, 372, 500])

    test_string_input_files_correct_image_with_convert_good_layout_test_string = (
        '{"image": {"files": ["something"],"type":"image","convert":{"shape":"[1,3,372,500]", "element_type":"float32", "layout":"NCHW"}}}'
    )

    def test_string_input_files_correct_image_with_convert(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_good_layout_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["convert"]["layout"], "NCHW")
        self.assertEqual(ret["image"]["convert"]["shape"], [1, 3, 372, 500])

    test_string_input_files_correct_image_with_convert_bad_layout_test_string = (
        '{"image": {"files": ["something"],"type":"image","convert":{"shape":"[1,3,372,500]", "element_type":"float32", "layout":["N","C","H","W"]}}}'
    )

    def test_string_input_files_correct_image_with_convert(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_bad_layout_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["convert"]["layout"], "NCHW")
        self.assertEqual(ret["image"]["convert"]["shape"], [1, 3, 372, 500])

    test_string_input_files_correct_image_without_convert_test_string = '{"image": {"files": ["something"],"type":"image"}}'

    def test_string_input_files_correct_image_without_convert(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_image_without_convert_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())

    test_string_input_files_correct_bin_test_string = '{"image": {"files": ["something"],"type":"bin","shape":"[1,3,372,500]", "element_type":"float32"}}'

    def test_string_input_files_correct_bin(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_bin_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["shape"], [1, 3, 372, 500])

    test_string_input_files_correct_bin_good_shape_test_string = (
        '{"image": {"files": ["something"],"type":"bin","shape":[1,3,372,500], "element_type":"float32"}}'
    )

    def test_string_input_files_correct_bin(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_bin_good_shape_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["shape"], [1, 3, 372, 500])

    test_string_input_files_correct_bin_bad_layout_test_string = (
        '{"image": {"files": ["something"],"type":"bin","shape":[1,3,372,500], "element_type":"float32", "layout":["N","C","H","W"]}}'
    )

    def test_string_input_files_correct_bin(self):
        test_string = UtilsTests_file_input_validation.test_string_input_files_correct_bin_bad_layout_test_string
        ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_string)
        self.assertTrue(len(ret.keys()) == 1)
        self.assertTrue("image" in ret.keys())
        self.assertEqual(ret["image"]["shape"], [1, 3, 372, 500])
        self.assertEqual(ret["image"]["layout"], "NCHW")

    test_string_input_files_bin_without_important_fields_test_string = {
        "shape": '{"image": {"files": ["something"],"type":"bin","!!shape!!":"[1,3,372,500]", "element_type":"float32"}}',
        "element_type": '{"image": {"files": ["something"],"type":"bin","shape":"[1,3,372,500]", "!!!!element_type!!!!!":"float32"}}',
    }

    def test_string_input_files_bin_without_important_fields(self):
        test_strings = UtilsTests_file_input_validation.test_string_input_files_bin_without_important_fields_test_string

        throw, error = UtilsTests_file_input_validation.get_result_test_string_input_files_without_important_fields(test_strings)
        self.assertTrue(throw, error)

    test_string_input_files_bin_with_forbidden_fields_test_string = {
        "convert": '{"image": {"files": ["something"],"type":"bin","shape":"[1,3,372,500]", "element_type":"float32","convert":{}}}'
    }

    def test_string_input_files_bin_with_forbidden_fields(self):
        test_strings = UtilsTests_file_input_validation.test_string_input_files_bin_with_forbidden_fields_test_string

        throw, error = UtilsTests_file_input_validation.get_result_test_string_input_files_without_important_fields(test_strings)
        self.assertTrue(throw, error)

    test_string_input_files_image_with_forbidden_fields_test_string = {
        "shape": '{"image": {"files": ["something"],"type":"image","shape":"[1,3,372,500]"}}',
        "element_type": '{"image": {"files": ["something"],"type":"image", "element_type":"float32"}}',
    }

    def test_string_input_files_image_with_forbidden_fields(self):
        test_strings = UtilsTests_file_input_validation.test_string_input_files_image_with_forbidden_fields_test_string

        throw, error = UtilsTests_file_input_validation.get_result_test_string_input_files_without_important_fields(test_strings)
        self.assertTrue(throw, error)


class UtilsTests_file_input_validation_from_file(unittest.TestCase):

    def create_positive_case_test_file_from_data(self, data, file_path):
        data_obj = json.loads(data)
        self.files_to_delete.append(file_path)
        with open(file_path, "w") as file:
            json.dump(data_obj, file)
        self.files_for_positive_case.append(file_path)

    def create_negative_case_test_file_from_data(self, dict_data, file_path_template):
        file_path_components = os.path.splitext(file_path_template)
        for expected_error_phrase, data in dict_data.items():
            data_obj = json.loads(data)
            file_path = file_path_components[0] + "_" + expected_error_phrase + file_path_components[1]
            self.files_to_delete.append(file_path)
            with open(file_path, "w") as file:
                json.dump(data_obj, file)
            self.files_for_negative_case[file_path] = expected_error_phrase

    def setUp(self):
        self.files_to_delete = []
        self.files_for_positive_case = []
        self.files_for_negative_case = {}
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_test_string,
            "test_string_input_files_correct_image_with_convert_test_string.json",
        )
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_good_shape_test_string,
            "test_string_input_files_correct_image_with_convert_good_shape_test_string.json",
        )
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_image_without_convert_test_string,
            "test_string_input_files_correct_image_without_convert_test_string.json",
        )
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_bin_test_string, "test_string_input_files_correct_bin_test_string.json"
        )
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_bin_bad_layout_test_string,
            "test_string_input_files_correct_bin_bad_layout_test_string.json",
        )
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_bin_good_shape_test_string,
            "test_string_input_files_correct_bin_good_shape_test_string.json",
        )

        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_good_layout_test_string,
            "test_string_input_files_correct_image_with_convert_good_layout_test_string.json",
        )
        self.create_positive_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_correct_image_with_convert_bad_layout_test_string,
            "test_string_input_files_correct_image_with_convert_bad_layout_test_string.json",
        )

        self.create_negative_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_bin_without_important_fields_test_string,
            "test_string_input_files_bin_without_important_fields_test_string.json",
        )
        self.create_negative_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_bin_with_forbidden_fields_test_string,
            "test_string_input_files_bin_with_forbidden_fields_test_string.json",
        )
        self.create_negative_case_test_file_from_data(
            UtilsTests_file_input_validation.test_string_input_files_image_with_forbidden_fields_test_string,
            "test_string_input_files_image_with_forbidden_fields_test_string.json",
        )

    def tearDown(self):
        for f in self.files_to_delete:
            if os.path.isfile(f):
                os.remove(f)

    def test_positive_json_loading_cases(self):
        for test_file in self.files_for_positive_case:
            ret = UtilsTests_file_input_validation.get_result_test_string_input_files_correct(test_file)
            self.assertTrue(len(ret.keys()) == 1)
            self.assertTrue("image" in ret.keys() or "bin" in ret.keys())

    def test_negative_json_loading_cases(self):
        for test_file, error in self.files_for_negative_case.items():
            throw, error = UtilsTests_file_input_validation.get_result_test_string_input_files_without_important_fields({error: test_file})
            self.assertTrue(throw, error)


if __name__ == "__main__":
    unittest.main()
