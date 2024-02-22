# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

from openvino.tools.mo.front.mxnet.multibox_detection_ext import MultiBoxDetectionOutputExtractor
from unit_tests.utils.extractors import PB


class TestMultiBoxDetection_Parsing(unittest.TestCase):
    def test_multi_box_detection_check_attrs(self):
        params = {'attrs': {
            "force_suppress": "True",
            "nms_threshold": "0.4",
            "nms_topk": "400",
            "variances": "(0.1, 0.1, 0.2, 0.2)"
        }}

        node = PB({'symbol_dict': params})
        MultiBoxDetectionOutputExtractor.extract(node)

        exp_attrs = {
            'type': 'DetectionOutput',
            'keep_top_k': 400,
            'variance_encoded_in_target': 0,
            'code_type': "caffe.PriorBoxParameter.CENTER_SIZE",
            'share_location': 1,
            'confidence_threshold': 0.01,
            'background_label_id': 0,
            'nms_threshold': 0.4,
            'top_k': 400,
            'decrease_label_id': 1,
            'clip_before_nms': 1,
            'normalized': 1,
        }

        for key in exp_attrs.keys():
            self.assertEqual(node[key], exp_attrs[key])

    def test_multi_box_detection_check_attrs_without_top_k(self):
        params = {'attrs': {
            "force_suppress": "True",
            "nms_threshold": "0.2",
            "threshold": "0.02",
            "variances": "(0.1, 0.1, 0.2, 0.2)"
        }}

        node = PB({'symbol_dict': params})
        MultiBoxDetectionOutputExtractor.extract(node)

        exp_attrs = {
            'type': 'DetectionOutput',
            'keep_top_k': -1,
            'variance_encoded_in_target': 0,
            'code_type': "caffe.PriorBoxParameter.CENTER_SIZE",
            'share_location': 1,
            'confidence_threshold': 0.02,
            'background_label_id': 0,
            'nms_threshold': 0.2,
            'top_k': -1,
            'decrease_label_id': 1,
            'clip_before_nms': 1,
            'normalized': 1,
        }

        for key in exp_attrs.keys():
            self.assertEqual(node[key], exp_attrs[key])
