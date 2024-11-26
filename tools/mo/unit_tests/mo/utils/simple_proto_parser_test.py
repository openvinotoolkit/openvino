# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
import unittest

from openvino.tools.mo.utils.simple_proto_parser import SimpleProtoParser

correct_proto_message_1 = 'model { faster_rcnn { num_classes: 90 image_resizer { keep_aspect_ratio_resizer {' \
                          ' min_dimension: 600  max_dimension: 1024 }}}}'

correct_proto_message_2 = '    first_stage_anchor_generator {grid_anchor_generator {height_stride: 16 width_stride:' \
                          ' 16 scales: 0.25 scales: 0.5 scales: 1.0 scales: 2.0  aspect_ratios: 0.5 aspect_ratios:' \
                          ' 1.0 aspect_ratios: 2.0}}'

correct_proto_message_3 = '  initializer \n{variance_scaling_initializer \n{\nfactor: 1.0 uniform: true bla: false ' \
                          'mode: FAN_AVG}}'

correct_proto_message_4 = 'train_input_reader {label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"' \
                          ' tf_record_input_reader { input_path: "PATH_TO_BE_CONFIGURED/  mscoco_train.record" }}'

correct_proto_message_5 = '  initializer \n  # abc \n{variance_scaling_initializer \n{\nfactor: 1.0 \n  # sd ' \
                          '\nuniform: true bla: false mode: FAN_AVG}}'

correct_proto_message_6 = '    first_stage_anchor_generator {grid_anchor_generator {height_stride: 16 width_stride:' \
                          ' 16 scales: [ 0.25, 0.5, 1.0, 2.0] aspect_ratios: 0.5 aspect_ratios:' \
                          ' 1.0 aspect_ratios: 2.0}}'

correct_proto_message_7 = '    first_stage_anchor_generator {grid_anchor_generator {height_stride: 16 width_stride:' \
                          ' 16 scales: [ 0.25, 0.5, 1.0, 2.0] aspect_ratios: [] }}'

correct_proto_message_8 = 'model {good_list: [3.0, 5.0, ]}'

correct_proto_message_9 = '    first_stage_anchor_generator {grid_anchor_generator {height_stride: 16, width_stride:' \
                          ' 16 scales: [ 0.25, 0.5, 1.0, 2.0], aspect_ratios: [] }}'

correct_proto_message_10 = r'train_input_reader {label_map_path: "C:\mscoco_label_map.pbtxt"' \
                           ' tf_record_input_reader { input_path: "PATH_TO_BE_CONFIGURED/  mscoco_train.record" }}'

correct_proto_message_11 = r'model {path: "C:\[{],}" other_value: [1, 2, 3, 4]}'

incorrect_proto_message_1 = 'model { bad_no_value }'

incorrect_proto_message_2 = 'model { abc: 3 { }'

incorrect_proto_message_3 = 'model { too_many_values: 3 4 }'

incorrect_proto_message_4 = 'model { missing_values: '

incorrect_proto_message_5 = 'model { missing_values: aa bb : }'

incorrect_proto_message_6 = 'model : '

incorrect_proto_message_7 = 'model : {bad_list: [3.0, 4, , 4.0]}'


class TestingSimpleProtoParser(unittest.TestCase):
    def test_correct_proto_reader_from_string_1(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_1)
        expected_result = {'model': {'faster_rcnn': {'num_classes': 90, 'image_resizer': {
            'keep_aspect_ratio_resizer': {'min_dimension': 600, 'max_dimension': 1024}}}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_2(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_2)
        expected_result = {'first_stage_anchor_generator': {
            'grid_anchor_generator': {'height_stride': 16, 'width_stride': 16, 'scales': [0.25, 0.5, 1.0, 2.0],
                                      'aspect_ratios': [0.5, 1.0, 2.0]}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_3(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_3)
        expected_result = {
            'initializer': {
                'variance_scaling_initializer': {'factor': 1.0, 'uniform': True, 'bla': False, 'mode': 'FAN_AVG'}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_4(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_4)
        expected_result = {
            'train_input_reader': {'label_map_path': "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt",
                                   'tf_record_input_reader': {
                                       'input_path': "PATH_TO_BE_CONFIGURED/  mscoco_train.record"}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_comments(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_5)
        expected_result = {
            'initializer': {
                'variance_scaling_initializer': {'factor': 1.0, 'uniform': True, 'bla': False, 'mode': 'FAN_AVG'}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_lists(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_6)
        expected_result = {'first_stage_anchor_generator': {
            'grid_anchor_generator': {'height_stride': 16, 'width_stride': 16, 'scales': [0.25, 0.5, 1.0, 2.0],
                                      'aspect_ratios': [0.5, 1.0, 2.0]}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_empty_list(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_7)
        expected_result = {'first_stage_anchor_generator': {
            'grid_anchor_generator': {'height_stride': 16, 'width_stride': 16, 'scales': [0.25, 0.5, 1.0, 2.0],
                                      'aspect_ratios': []}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_comma_trailing_list(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_8)
        expected_result = {'model': {'good_list': [3.0, 5.0]}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_redundant_commas(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_9)
        expected_result = {'first_stage_anchor_generator': {
            'grid_anchor_generator': {'height_stride': 16, 'width_stride': 16, 'scales': [0.25, 0.5, 1.0, 2.0],
                                      'aspect_ratios': []}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_windows_path(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_10)
        expected_result = {
            'train_input_reader': {'label_map_path': r"C:\mscoco_label_map.pbtxt",
                                   'tf_record_input_reader': {
                                       'input_path': "PATH_TO_BE_CONFIGURED/  mscoco_train.record"}}}
        self.assertDictEqual(result, expected_result)

    def test_correct_proto_reader_from_string_with_special_characters_in_string(self):
        result = SimpleProtoParser().parse_from_string(correct_proto_message_11)
        expected_result = {'model': {'path': r"C:\[{],}",
                                     'other_value': [1, 2, 3, 4]}}
        self.assertDictEqual(result, expected_result)
    
    @unittest.skip
    def test_incorrect_proto_reader_from_string_1(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_1)
        self.assertIsNone(result)

    def test_incorrect_proto_reader_from_string_2(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_2)
        self.assertIsNone(result)

    def test_incorrect_proto_reader_from_string_3(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_3)
        self.assertIsNone(result)

    def test_incorrect_proto_reader_from_string_4(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_4)
        self.assertIsNone(result)

    def test_incorrect_proto_reader_from_string_5(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_5)
        self.assertIsNone(result)

    def test_incorrect_proto_reader_from_string_6(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_6)
        self.assertIsNone(result)

    def test_incorrect_proto_reader_from_string_7(self):
        result = SimpleProtoParser().parse_from_string(incorrect_proto_message_7)
        self.assertIsNone(result)

    def test_correct_proto_reader_from_file(self):
        file = tempfile.NamedTemporaryFile('wt', delete=False)
        file.write(correct_proto_message_1)
        file_name = file.name
        file.close()

        result = SimpleProtoParser().parse_file(file_name)
        expected_result = {'model': {'faster_rcnn': {'num_classes': 90, 'image_resizer': {
            'keep_aspect_ratio_resizer': {'min_dimension': 600, 'max_dimension': 1024}}}}}
        self.assertDictEqual(result, expected_result)
        os.unlink(file_name)

    @unittest.skip("Temporary disabled since chmod() is temporary not working on Linux. (Windows do not support not writable dir at all)")
    def test_proto_reader_from_non_readable_file(self):
        file = tempfile.NamedTemporaryFile('wt', delete=False)
        file.write(correct_proto_message_1)
        file_name = file.name
        file.close()
        os.chmod(file_name, 0000)

        result = SimpleProtoParser().parse_file(file_name)
        self.assertIsNone(result)
        os.unlink(file_name)

    def test_proto_reader_from_non_existing_file(self):
        result = SimpleProtoParser().parse_file('/non/existing/file')
        self.assertIsNone(result)
