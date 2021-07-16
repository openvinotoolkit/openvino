# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unittest

import numpy as np
from generator import generator, generate

from mo.utils.error import Error
from mo.utils.utils import match_shapes, validate_json_config, get_mo_root_dir


class TestMatchShapes(unittest.TestCase):

    def run_match_shapes(self, pattern: list, shape: list):
        return match_shapes(np.array(pattern, dtype=np.int64), np.array(shape, dtype=np.int64))

    def test_positive(self):
        self.assertTrue(self.run_match_shapes([], []))
        self.assertTrue(self.run_match_shapes([1,2,3], [1,2,3]))
        self.assertTrue(self.run_match_shapes([-1,2,3], [1,2,3]))
        self.assertTrue(self.run_match_shapes([1,-1,3], [1,2,3]))
        self.assertTrue(self.run_match_shapes([-1,-1,-1], [1,2,3]))
        self.assertTrue(self.run_match_shapes([-1], [2]))

    def test_negative(self):
        self.assertFalse(self.run_match_shapes([-1], []))
        self.assertFalse(self.run_match_shapes([-1], [1,2,3]))
        self.assertFalse(self.run_match_shapes([-1,2,3], [1,3,3]))
        self.assertFalse(self.run_match_shapes([1,-1,3], [2,2]))
        self.assertFalse(self.run_match_shapes([-1, -1, -1], [2, 3, 4, 5]))




@generator
class TestSchema(unittest.TestCase):
    base_dir = get_mo_root_dir()
    path = os.path.join(base_dir, 'extensions', 'front', )
    schema_file = os.path.join(base_dir, 'mo', 'utils', 'schema.json')

    test_json1 = '[{"id": "", "match_kind": "general", "custom_attributes": {}}]'
    test_json2 = '[{"id": "someid", "match_kind": "abc", "custom_attributes": {}}]'

    @generate(*[('tf', 'efficient_det_support_api_v2.0.json'),
                ('tf', 'efficient_det_support_api_v2.4.json'),
                ('tf', 'faster_rcnn_support_api_v1.7.json'),
                ('tf', 'faster_rcnn_support_api_v1.10.json'),
                ('tf', 'faster_rcnn_support_api_v1.13.json'),
                ('tf', 'faster_rcnn_support_api_v1.14.json'),
                ('tf', 'faster_rcnn_support_api_v1.15.json'),
                ('tf', 'faster_rcnn_support_api_v2.0.json'),
                ('tf', 'faster_rcnn_support_api_v2.4.json'),
                ('tf', 'mask_rcnn_support.json'),
                ('tf', 'mask_rcnn_support_api_v1.7.json'),
                ('tf', 'mask_rcnn_support_api_v1.11.json'),
                ('tf', 'mask_rcnn_support_api_v1.13.json'),
                ('tf', 'mask_rcnn_support_api_v1.14.json'),
                ('tf', 'mask_rcnn_support_api_v1.15.json'),
                ('tf', 'mask_rcnn_support_api_v2.0.json'),
                ('tf', 'retinanet.json'),
                ('tf', 'rfcn_support.json'),
                ('tf', 'rfcn_support_api_v1.10.json'),
                ('tf', 'rfcn_support_api_v1.13.json'),
                ('tf', 'rfcn_support_api_v1.14.json'),
                ('tf', 'ssd_support.json'),
                ('tf', 'ssd_support_api_v1.14.json'),
                ('tf', 'ssd_support_api_v1.15.json'),
                ('tf', 'ssd_support_api_v2.0.json'),
                ('tf', 'ssd_support_api_v2.4.json'),
                ('tf', 'ssd_toolbox_detection_output.json'),
                ('tf', 'ssd_toolbox_multihead_detection_output.json'),
                ('tf', 'ssd_v2_support.json'),
                ('tf', 'yolo_v1.json'),
                ('tf', 'yolo_v1_tiny.json'),
                ('tf', 'yolo_v2.json'),
                ('tf', 'yolo_v2_tiny.json'),
                ('tf', 'yolo_v2_tiny_voc.json'),
                ('tf', 'yolo_v3.json'),
                ('tf', 'yolo_v3_tiny.json'),
                ('tf', 'yolo_v3_voc.json'),
                ('onnx', 'faster_rcnn.json'),
                ('onnx', 'person_detection_crossroad.json'),
                ('mxnet', 'yolo_v3_mobilenet1_voc.json'),
                ])
    def test_schema_file(self, config_path, transformation_config):
        transformation_file = os.path.join(self.path, config_path, transformation_config)
        with open(transformation_file, 'r') as f:
            data = json.load(f)
            validate_json_config(data, transformation_file)

    def test_schema_id_empty(self):
        data = json.loads(self.test_json1)
        self.assertRaises(Error, validate_json_config, data, self.test_json1)

    def test_schema_match_kind_wrong(self):
        data = json.loads(self.test_json2)
        self.assertRaises(Error, validate_json_config, data, self.test_json2)
