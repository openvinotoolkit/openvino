# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest

rng = np.random.default_rng(233534)


class TestKerasDot(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['x'] = rng.uniform(-2.0, 2.0, x_shape).astype(self.input_type)
            inputs_data['y'] = rng.uniform(-2.0, 2.0, y_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data['x'] = rng.integers(-3, 3, x_shape).astype(self.input_type)
            inputs_data['y'] = rng.integers(-3, 3, y_shape).astype(self.input_type)
        else:
            inputs_data['x'] = rng.integers(0, 3, x_shape).astype(self.input_type)
            inputs_data['y'] = rng.integers(0, 3, y_shape).astype(self.input_type)
        return inputs_data

    def create_keras_dot_net(self, input_shapes, axes, input_type, normalize):
        self.input_type = input_type
        tf.keras.backend.clear_session()
        inputs = [tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type, name='x'),
                  tf.keras.Input(shape=input_shapes[1][1:], dtype=input_type, name='y')]
        y = tf.keras.layers.Dot(axes, normalize=normalize)(inputs)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        return tf2_net, None

    @pytest.mark.parametrize('input_shapes,axes', [
        ([[5, 4], [5, 4]], 1),
        ([[5, 1, 4], [5, 1, 4]], 1),
        ([[5, 1, 4], [5, 1, 4]], 2),
        ([[5, 1, 5, 2], [5, 1, 2, 5]], (2, 3)),
        ([[5, 1, 102, 2], [5, 1, 4, 102]], (-2, -1))
    ])
    @pytest.mark.parametrize('input_type', [np.float32, np.float64])
    @pytest.mark.parametrize('normalize', [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_dot(self, input_shapes, axes, input_type, normalize,
                       ie_device, precision, temp_dir, ir_version,
                       use_legacy_frontend):
        params = {}
        params['input_shapes'] = input_shapes
        self._test(*self.create_keras_dot_net(input_shapes, axes, input_type, normalize),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
