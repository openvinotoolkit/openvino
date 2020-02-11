"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import onnx
import unittest

import numpy as np

from extensions.front.onnx.detection_output import DetectionOutputFrontExtractor
from extensions.ops.DetectionOutput import DetectionOutput
from mo.ops.op import Op
from mo.utils.unittest.extractors import PB


class TestDetectionOutputExt(unittest.TestCase):
    @staticmethod
    def _create_do_node(num_classes=0, share_location=0, background_label_id=0,
                        code_type="", variance_encoded_in_target=0, keep_top_k=0,
                        confidence_threshold=0, nms_threshold=0, top_k=0, eta=0):
        pb = onnx.helper.make_node(
            'DetectionOutput',
            inputs=['x'],
            outputs=['y'],
            num_classes=num_classes,
            share_location=share_location,
            background_label_id=background_label_id,
            code_type=code_type,
            variance_encoded_in_target=variance_encoded_in_target,
            keep_top_k=keep_top_k,
            confidence_threshold=confidence_threshold,
            # nms_param
            nms_threshold=nms_threshold,
            top_k=top_k,
            eta=eta,
        )
        
        node = PB({'pb': pb})
        return node

    @classmethod
    def setUpClass(cls):
        Op.registered_ops['DetectionOutput'] = DetectionOutput

    def test_do_no_pb_no_ml(self):
        self.assertRaises(AttributeError, DetectionOutputFrontExtractor.extract, None)

    def test_do_ext_ideal_numbers(self):
        node = self._create_do_node(num_classes=21, share_location=1,
                                    code_type="CENTER_SIZE", keep_top_k=200,
                                    confidence_threshold=0.01, nms_threshold=0.45, top_k=400, eta=1.0)
        
        DetectionOutputFrontExtractor.extract(node)
        
        exp_res = {
            'op': 'DetectionOutput',
            'type': 'DetectionOutput',
            'num_classes': 21,
            'share_location': 1,
            'background_label_id': 0,
            'code_type': "caffe.PriorBoxParameter.CENTER_SIZE",
            'variance_encoded_in_target': 0,
            'keep_top_k': 200,
            'confidence_threshold': 0.01,
            'visualize_threshold': 0.6,
            # nms_param
            'nms_threshold': 0.45,
            'top_k': 400,
            'eta': 1.0,
            # ONNX have not such parameters
            # save_output_param.resize_param
            'prob': 0,
            'resize_mode': "",
            'height': 0,
            'width': 0,
            'height_scale': 0,
            'width_scale': 0,
            'pad_mode': "",
            'pad_value': "",
            'interp_mode': "",
            'input_width': 1,
            'input_height': 1,
            'normalized': 1,            
        }

        for key in exp_res.keys():
            if key in ['confidence_threshold', 'visualise_threshold', 'nms_threshold', 'eta']:
                np.testing.assert_almost_equal(node[key], exp_res[key])
            else:
                self.assertEqual(node[key], exp_res[key])
