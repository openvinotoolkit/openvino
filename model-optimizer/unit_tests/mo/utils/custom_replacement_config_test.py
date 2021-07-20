# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from generator import generator, generate

from mo.utils.custom_replacement_config import load_and_validate_json_config
from mo.utils.error import Error
from mo.utils.utils import get_mo_root_dir


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
        self.assertTrue(load_and_validate_json_config(transformation_file))

    def test_schema_id_empty(self):
        self.assertRaises(Error, load_and_validate_json_config, self.test_json1)

    def test_schema_match_kind_wrong(self):
        self.assertRaises(Error, load_and_validate_json_config, self.test_json2)
