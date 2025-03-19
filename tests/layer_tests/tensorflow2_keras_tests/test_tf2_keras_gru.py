# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest

rng = np.random.default_rng(233534)


class TestKerasGru(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x']
        inputs_data = {}
        inputs_data['x'] = rng.uniform(-2.0, 2.0, x_shape).astype(self.input_type)
        return inputs_data

    def create_keras_gru_net(self, input_shapes, input_type, units,
                             activation, recurrent_activation,
                             dropouts, use_bias, flag1, flag2):
        self.input_type = input_type
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type, name='x')
        dropout, recurrent_dropout = dropouts
        go_backwards, reset_after = flag1, flag2
        y = tf.keras.layers.GRU(units=units, activation=activation,
                                recurrent_activation=recurrent_activation,
                                use_bias=use_bias, dropout=dropout,
                                recurrent_dropout=recurrent_dropout,
                                return_sequences=False, return_state=False,
                                go_backwards=go_backwards, reset_after=reset_after)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])
        ref_net = None

        return tf2_net, ref_net

    @pytest.mark.parametrize('input_shapes', [[[2, 3, 4]]])
    @pytest.mark.parametrize('input_type', [np.float32, np.float64])
    @pytest.mark.parametrize('units', [1, 2, 3])
    @pytest.mark.parametrize('activation', ['tanh', 'relu', 'elu', 'linear'])
    @pytest.mark.parametrize('recurrent_activation', ['sigmoid', 'tanh', 'softmax'])
    @pytest.mark.parametrize('dropouts', [(.0, .0), (.0, .3), (.2, .4), ])
    @pytest.mark.parametrize('use_bias', [True, False])
    @pytest.mark.parametrize('flag1', [True, False])
    @pytest.mark.parametrize('flag2', [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_gru(self, input_shapes, input_type, units,
                       activation, recurrent_activation,
                       dropouts, use_bias, flag1, flag2,
                       ie_device, precision, temp_dir, ir_version,
                       use_legacy_frontend):
        params = {}
        params['input_shapes'] = input_shapes
        self._test(*self.create_keras_gru_net(input_shapes, input_type, units,
                                              activation, recurrent_activation,
                                              dropouts, use_bias, flag1, flag2),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
