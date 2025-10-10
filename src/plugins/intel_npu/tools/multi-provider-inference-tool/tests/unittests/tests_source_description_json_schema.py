#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import json
import jsonschema
import os
import sys
import unittest

sys.path.append("../..")
import common.source_description_schema

from common.enums import InputSourceFileType


class UtilsTests_source_description_schema_for_image(unittest.TestCase):
    def test_valid_source_no_convert(self):
        valid_input = {"files": ["image_path_0", "image_path_1"], "type": "image", "allowed_unexpected_field": 4}

        obj = None
        try:
            obj = common.source_description_schema.InputSource(valid_input)
        except Exception:
            self.fail("test_valid_source_no_convert() raised Exception unexpectedly")

        self.assertTrue("type" in obj.keys())
        self.assertEqual(obj["type"], InputSourceFileType.image.name)

    def test_valid_source_with_convert(self):
        valid_input_with_convert = {
            "files": ["image_path_0", "image_path_1"],
            "type": "image",
            "convert": {"shape": [4, 4], "layout": "something"},
            "allowed_unexpected_field": 4,
        }

        obj = None
        try:
            obj = common.source_description_schema.InputSource(valid_input_with_convert)
        except Exception:
            self.fail("test_valid_source_with_convert() raised Exception unexpectedly")

        self.assertTrue("type" in obj)
        self.assertEqual(obj["type"], InputSourceFileType.image.name)

    def test_fail_source_with_convert(self):
        valid_input_with_convert_fail = {
            "files": ["image_path_0", "image_path_1"],
            "type": "image",
            "convert": {"shape": [4, 4], "layout": "something", "unexpected_field": "abc"},
            "allowed_unexpected_field": 4,
        }

        with self.assertRaises(RuntimeError):
            common.source_description_schema.InputSource(valid_input_with_convert_fail)

    def test_fail_source_with_forbidden_fields(self):
        valid_input_with_forbidden_fail = {
            "files": ["image_path_0", "image_path_1"],
            "type": "image",
            "convert": {"shape": [4, 4], "layout": "something"},
            "shape": "shape is forbidden",
            "element_type": "element_type is forbidden",
            "allowed_unexpected_field": 4,
        }

        with self.assertRaises(RuntimeError):
            common.source_description_schema.InputSource(valid_input_with_forbidden_fail)


class UtilsTests_source_description_schema_for_bin(unittest.TestCase):
    def test_valid_source_default_type(self):
        valid_input = {
            "files": ["image_path_0", "image_path_1"],
            # presumed  "type":"bin",
            "shape": "[1,3,372,500]",
            "element_type": "abc",
            "allowed_unexpected_field": 4,
        }

        obj = None
        try:
            obj = common.source_description_schema.InputSource(valid_input)
        except Exception:
            self.fail("test_valid_source_default_type() raised Exception unexpectedly")

        self.assertTrue("type" in obj)
        self.assertEqual(obj["type"], InputSourceFileType.bin.name)

    def test_fail_source_no_required_fields(self):
        no_element_type_input = {"files": ["image_path_0", "image_path_1"], "shape": "[1,3,372,500]", "allowed_unexpected_field": 4}

        with self.assertRaises(RuntimeError):
            common.source_description_schema.InputSource(no_element_type_input)

        shape_type_input = {"files": ["image_path_0", "image_path_1"], "element_type": "abc", "allowed_unexpected_field": 4}

        with self.assertRaises(RuntimeError):
            common.source_description_schema.InputSource(shape_type_input)

    def test_fail_source_with_forbidden_fields(self):
        valid_input_with_forbidden_fail = {
            "files": ["image_path_0", "image_path_1"],
            "type": "bin",
            "convert": "convert is forbidden regardless its type",
            "shape": "[1,3,372,500]",
            "element_type": "abc",
            "allowed_unexpected_field": 4,
        }


if __name__ == "__main__":
    unittest.main()
