# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tf_layer_test_class import CommonTFLayerTest


class TestNonMaxSupression(CommonTFLayerTest):

    # overload inputs generation to suit NMS use case
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.uniform(low=0, high=1,
                                                   size=inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_nms_net(self, test_params: dict):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:

            # parametrized inputs
            number_of_boxes = test_params["number_of_boxes"]
            max_output_size = tf.constant(test_params["max_output_size"])
            iou_threshold = tf.constant(test_params["iou_threshold"])
            score_threshold = tf.constant(test_params["score_threshold"])

            # inputs to be generated
            boxes = tf.compat.v1.placeholder(tf.float32, [number_of_boxes, 4], "Boxes")

            # randomize boxes' confidence scores
            np.random.seed(42)
            scores = np.random.uniform(low=0.2, high=1.0, size=[number_of_boxes])

            _ = tf.image.non_max_suppression(boxes, scores, max_output_size,
                                             iou_threshold, score_threshold, name="NMS")
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    def create_nms_net_with_scores(self, test_params: dict):

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:

            # parametrized inputs
            number_of_boxes = test_params["number_of_boxes"]
            max_output_size = tf.constant(test_params["max_output_size"])
            iou_threshold = tf.constant(test_params["iou_threshold"])
            score_threshold = tf.constant(test_params["score_threshold"])
            soft_nms_sigma = tf.constant(test_params["soft_nms_sigma"])

            # inputs to be generated
            boxes = tf.compat.v1.placeholder(tf.float32, [number_of_boxes, 4], "Boxes")

            # randomize boxes' confidence scores
            np.random.seed(42)
            scores = np.random.uniform(low=0.2, high=1.0, size=[number_of_boxes])

            _ = tf.image.non_max_suppression_with_scores(boxes, scores, max_output_size,
                                             iou_threshold, score_threshold, soft_nms_sigma, name="NMS")
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_params = [
            (
                {
                    "number_of_boxes": 50,
                    "max_output_size": 5,
                    "iou_threshold": 0.7,
                    "score_threshold": 0.8,
                    "soft_nms_sigma": 0.1
                }
            ),
            (
                {
                    "number_of_boxes": 50,
                    "max_output_size": 9,
                    "iou_threshold": 0.7,
                    "score_threshold": 0.7,
                    "soft_nms_sigma": 0.4
                }
            ),
            (
                {
                    "number_of_boxes": 50,
                    "max_output_size": 3,
                    "iou_threshold": 0.3,
                    "score_threshold": 0.8,
                    "soft_nms_sigma": 0.7
                }
            )
        ]

    @pytest.mark.parametrize("test_params", test_params)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_NonMaxSupression(self, test_params, ie_device, precision, ir_version, temp_dir,
                              use_new_frontend, api_2):
        self._test(*self.create_nms_net(test_params), ie_device, precision,
                   ir_version, temp_dir, api_2, use_new_frontend)

    @pytest.mark.parametrize("test_params", test_params)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_NonMaxSupressionWithScores(self, test_params, ie_device, precision, ir_version, temp_dir,
                                        use_new_frontend, api_2):
        self._test(*self.create_nms_net_with_scores(test_params), ie_device, precision,
                   ir_version, temp_dir, api_2, use_new_frontend)
