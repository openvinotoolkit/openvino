# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest.mock

from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.pipeline_config import PipelineConfig

file_content = """model {
  faster_rcnn {
    num_classes: 90
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
        pad_to_max_dimension: true
      }
    }
    feature_extractor {
      type: "faster_rcnn_inception_v2"
      first_stage_features_stride: 16
    }
    first_stage_anchor_generator {
      grid_anchor_generator {
        height_stride: 16
        width_stride: 16
        scales: 0.25
        scales: 0.5
        scales: 1.0
        scales: 2.0
        aspect_ratios: 0.5
        aspect_ratios: 1.0
        aspect_ratios: 2.0
      }
    }
    first_stage_box_predictor_conv_hyperparams {
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.00999999977648
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.699999988079
    first_stage_max_proposals: 100
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
        use_dropout: false
        dropout_keep_probability: 1.0
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.300000011921
        iou_threshold: 0.600000023842
        max_detections_per_class: 100
        max_total_detections: 200
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}
"""


class TestingSimpleProtoParser(unittest.TestCase):
    def test_pipeline_config_not_existing_file(self):
        self.assertRaises(Error, PipelineConfig, "/abc/def")

    def test_pipeline_config_non_model_file(self):
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data="non_model {}")):
            self.assertRaises(Error, PipelineConfig, __file__)

    def test_pipeline_config_existing_file(self):
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open(read_data=file_content)):
            pipeline_config = PipelineConfig(__file__)
        expected_result = {'resizer_min_dimension': 600,
                           'first_stage_nms_score_threshold': 0.0,
                           'anchor_generator_aspect_ratios': [0.5, 1.0, 2.0],
                           'num_classes': 90,
                           'anchor_generator_scales': [0.25, 0.5, 1.0, 2.0],
                           'first_stage_max_proposals': 100,
                           'first_stage_nms_iou_threshold': 0.699999988079,
                           'resizer_max_dimension': 1024,
                           'initial_crop_size': 14,
                           'frcnn_variance_height': 5.0,
                           'frcnn_variance_width': 5.0,
                           'frcnn_variance_x': 10.0,
                           'frcnn_variance_y': 10.0,
                           'ssd_anchor_generator_base_anchor_width': 1.0,
                           'ssd_anchor_generator_base_anchor_height': 1.0,
                           'anchor_generator_height': 256,
                           'anchor_generator_width': 256,
                           'anchor_generator_height_stride': 16,
                           'anchor_generator_width_stride': 16,
                           'ssd_anchor_generator_min_scale': 0.2,
                           'ssd_anchor_generator_max_scale': 0.95,
                           'ssd_anchor_generator_interpolated_scale_aspect_ratio': 1.0,
                           'use_matmul_crop_and_resize': False,
                           'add_background_class': True,
                           'share_box_across_classes': False,
                           'pad_to_max_dimension': True,
                           'postprocessing_score_threshold': 0.300000011921,
                           'postprocessing_score_converter': 'SOFTMAX',
                           'postprocessing_iou_threshold': 0.600000023842,
                           'postprocessing_max_detections_per_class': 100,
                           'postprocessing_max_total_detections': 200,
                           }
        self.assertDictEqual(pipeline_config._model_params, expected_result)
