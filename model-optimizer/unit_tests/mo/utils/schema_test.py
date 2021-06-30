# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unittest

import fastjsonschema as json_validate
from generator import generator, generate

path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'extensions', 'front',)
schema_file = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'mo', 'utils', 'schema.json')

@generator
class TestSchema(unittest.TestCase):
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
        transformation_file = os.path.join(path, config_path, transformation_config)
        with open(transformation_file, 'r') as f:
            data = json.load(f)

        with open(schema_file, 'r') as f:
            schema = json.load(f)
            validator = json_validate.compile(schema)
            validator(data)
