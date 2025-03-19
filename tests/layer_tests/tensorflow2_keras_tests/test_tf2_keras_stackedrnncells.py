# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest

rng = np.random.default_rng(23423556)


class TestKerasStackedRNNCells(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        input_names = list(inputs_info.keys())
        assert len(input_names) == 1, "Test expects only one input"
        x_shape = inputs_info[input_names[0]]
        inputs_data = {}
        inputs_data[input_names[0]] = rng.uniform(-1.0, 1.0, x_shape).astype(np.float32)
        return inputs_data

    def create_keras_stackedrnncells_net(self, input_names, input_shapes, input_type, rnn_cells,
                                         ir_version):
        cells_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.keras.layers.<cell> operation have no "==" operation to be compared
            "LSTMCell": [tf.keras.layers.LSTMCell(128) for _ in range(2)],
            "GRUCell": [tf.keras.layers.GRUCell(50) for _ in range(3)]
        }
        rnn_cells = cells_structure[rnn_cells]

        # create TensorFlow 2 model with Keras StackedRNNCells operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        y = tf.keras.layers.RNN(stacked_lstm)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 3]], input_type=tf.float32,
             rnn_cells="LSTMCell"),
        dict(input_names=["x1"], input_shapes=[[5, 4, 3]], input_type=tf.float32,
             rnn_cells="GRUCell")
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_stackedrnncells(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        self._test(*self.create_keras_stackedrnncells_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
