# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest

rng = np.random.default_rng()


class TestKerasUpSampling2D(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['x'] = rng.uniform(-2.0, 2.0, x_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data['x'] = rng.integers(-8, 8, x_shape).astype(self.input_type)
        else:
            inputs_data['x'] = rng.integers(0, 8, x_shape).astype(self.input_type)
        return inputs_data

    def create_keras_upsampling2d_net(self, input_shapes, input_type, size,
                                      data_format, interpolation,
                                      ir_version):
        # create TensorFlow 2 model with Keras UpSampling2D operation
        self.input_type = input_type
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name='x', dtype=input_type)
        y = tf.keras.layers.UpSampling2D(size=size, data_format=data_format,
                                         interpolation=interpolation)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_nearest = [
        dict(input_shapes=[[5, 4, 8, 2]], size=(2, 3)),
        dict(input_shapes=[[3, 2, 4, 2]], size=(3, 1)),
        dict(input_shapes=[[1, 3, 8, 6]], size=(5, 2)),
    ]

    @pytest.mark.parametrize("params", test_data_nearest)
    @pytest.mark.parametrize("data_format", ['channels_last', 'channels_first'])
    @pytest.mark.parametrize("interpolation", ['nearest', 'bilinear'])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64, np.int8, np.uint8,
                                            np.int16, np.uint16, np.int32, np.int64])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_keras_upsampling2d_nearest(self, params, input_type, data_format, interpolation,
                                        ie_device, precision, ir_version, temp_dir,
                                        use_old_api, use_new_frontend):
        self._test(*self.create_keras_upsampling2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    # Tests for bilinear interpolation
    test_data_bilinear = [
        pytest.param((["x1"], [[1, 6, 2, 1]], tf.float32, (3, 1), 'channels_last', 'bilinear'), marks=pytest.mark.precommit_tf_fe),
        (["x1"], [[1, 3, 1, 6]], tf.float32, (5, 2), 'channels_last', 'bilinear')
    ]

    @pytest.mark.parametrize("params", test_data_bilinear)
    @pytest.mark.nightly
    def test_keras_upsampling2d_bilinear(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_old_api, use_new_frontend):
        input_names, input_shapes, input_type, size, data_format, interpolation = params
        self._test(*self.create_keras_upsampling2d_net(input_names=input_names, input_shapes=input_shapes, input_type=input_type, size=size, data_format=data_format, interpolation=interpolation,
        ir_version=ir_version), ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version, use_new_frontend=use_new_frontend, **params)

    test_data_channels_first = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 5, 3]], input_type=tf.float32,
             size=(3, 4), data_format='channels_first', interpolation='nearest'),
        dict(input_names=["x1"], input_shapes=[[3, 2, 7, 2]], input_type=tf.float32,
             size=(2, 3), data_format='channels_first', interpolation='nearest'),
        dict(input_names=["x1"], input_shapes=[[3, 5, 4, 6]], input_type=tf.float32,
             size=(5, 2), data_format='channels_first', interpolation='nearest'),
    ]

    @pytest.mark.parametrize("params", test_data_channels_first)
    @pytest.mark.nightly
    def test_keras_upsampling2d_channels_first(self, params, ie_device, precision, ir_version,
                                               temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_keras_upsampling2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
