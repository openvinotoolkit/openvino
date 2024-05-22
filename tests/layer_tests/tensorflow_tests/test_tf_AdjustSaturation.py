# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAdjustSaturation(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        images_shape = inputs_info['images:0']
        inputs_data = {}
        inputs_data['images:0'] = np.random.rand(*images_shape).astype(self.input_type)
        inputs_data['scale:0'] = np.random.rand()
        
        # inputs_data['images:0'] = np.array([
        #         [[0.4, 0.2, 0.3],
        #         [0.9, 0.5, 0.6]]
        #         ]).astype(np.float32)
        # inputs_data['scale:0'] = 2.1
        # hsv = rgb_to_hsv( inputs_data['images:0'])
        # hsv[...,1] =  np.clip(hsv[...,1]*inputs_data['scale:0'], 0.0, 1.0)
        # rgb = hsv_to_rgb(hsv)
        # logger.info(hsv)
        # logger.info(rgb)
        
        return inputs_data

    def create_adjust_saturation_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(input_type, input_shape, 'images')
            scale = tf.compat.v1.placeholder(input_type, [], 'scale')
            tf.raw_ops.AdjustSaturation(images=images, scale=scale)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    # Each input is a tensor of at least 3 dimensions. 
    # The last dimension is interpreted as channels, and must be three.
    test_data_basic = [
        dict(input_shape=[5, 5, 3], input_type=np.float32),
        dict(input_shape=[2, 3, 4, 3], input_type=np.float32),
        dict(input_shape=[1, 2, 3, 3, 3], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_adjust_saturation_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_adjust_saturation_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

def rgb_to_hsv(image):
    # Extract RGB channels
    r, g, b = image[..., 0], image[..., 1], image[..., 2]

    # Compute V, the maximum of R, G, B
    vv = np.max(image, axis=-1)

    # Compute the minimum of R, G, B
    min_rgb = np.min(image, axis=-1)

    # Compute the range as V - min(R, G, B)
    range = vv - min_rgb

    # logger.info("range")
    # logger.info(range)
    # Avoid division by zero
    s = np.where(vv == 0, 0, range / vv)
    
    # Normalize the hue to 0-1 scale and handle different cases
    norm = 1.0 / (6.0 * range + 1e-10)  # Adding a small epsilon to avoid division by zero

    hh = np.zeros_like(vv)
    hh = np.where(r == vv, norm * (g - b), hh)
    hh = np.where(g == vv, norm * (b - r) + 2.0 / 6.0, hh)
    hh = np.where(b == vv, norm * (r - g) + 4.0 / 6.0, hh)

    # Set hue to 0 where range is zero
    hh = np.where(range <= 0, 0, hh)

    hh = np.where(hh < 0, hh + 1, hh)

    # Return the HSV image
    hsv_image = np.stack([hh, s, vv], axis=-1)
    return hsv_image
def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    
    c = s * v
    m = v - c
    dh = h * 6
    h_category = np.floor(dh).astype(int)
    fmodu = dh % 2.0  # Simplified way to ensure fmodu is within [0, 2)

    x = c * (1 - np.abs(fmodu - 1))
    # logger.info(fmodu)

    # Initialize output arrays
    rr, gg, bb = np.zeros_like(c), np.zeros_like(c), np.zeros_like(c)

    # Conditions to set the correct values based on the h_category
    rr[(h_category == 0) | (h_category == 5)] = c[(h_category == 0) | (h_category == 5)]
    gg[(h_category == 1) | (h_category == 2)] = c[(h_category == 1) | (h_category == 2)]
    bb[(h_category == 3) | (h_category == 4)] = c[(h_category == 3) | (h_category == 4)]

    rr[(h_category == 1) | (h_category == 4)] = x[(h_category == 1) | (h_category == 4)]
    gg[(h_category == 0) | (h_category == 3)] = x[(h_category == 0) | (h_category == 3)]
    bb[(h_category == 2) | (h_category == 5)] = x[(h_category == 2) | (h_category == 5)]

    # Set the remaining channel for each category to 0, already initialized

    r = rr + m
    g = gg + m
    b = bb + m
    
    return np.stack([r, g, b], axis=-1)