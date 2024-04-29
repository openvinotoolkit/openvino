# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasRNN(CommonTF2LayerTest):
    def create_keras_rnn_net(self, input_names, input_shapes,
                             cell, return_sequences, return_state, go_backwards,
                             stateful, unroll, time_major,
                             ir_version):
        cells_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.keras.layers.<cell> operation have no "==" operation to be compared
            "LSTMCell": tf.keras.layers.LSTMCell(6),
            "GRUCell": tf.keras.layers.GRUCell(3),
            "SimpleRNNCell": tf.keras.layers.SimpleRNNCell(5)
        }
        cell = cells_structure[cell]

        # create TensorFlow 2 model with Keras RNN operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.RNN(cell=cell, return_sequences=return_sequences,
                                return_state=return_state,
                                go_backwards=go_backwards, stateful=stateful, unroll=unroll,
                                time_major=time_major
                                )(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    # Tests for default parameter values
    test_data = [
        dict(input_names=["x1"], input_shapes=[[2, 4, 5]],
             cell="LSTMCell", return_sequences=False,
             return_state=False, go_backwards=False,
             stateful=False, unroll=False, time_major=False,
             ),
        dict(input_names=["x1"], input_shapes=[[1, 3, 2]],
             cell="GRUCell", return_sequences=False,
             return_state=False, go_backwards=False,
             stateful=False, unroll=False, time_major=False,
             ),
        dict(input_names=["x1"], input_shapes=[[3, 2, 1]],
             cell="SimpleRNNCell", return_sequences=False,
             return_state=False, go_backwards=False,
             stateful=False, unroll=False, time_major=False,
             )
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_rnn(self, params, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        self._test(*self.create_keras_rnn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for default parameter values
    test_data_multiple_outputs = [
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[2, 4, 5]],
                 cell="LSTMCell", return_sequences=True,
                 return_state=True, go_backwards=False,
                 stateful=False, unroll=False, time_major=False),
            marks=pytest.mark.xfail(reason="49537")),
        dict(input_names=["x1"], input_shapes=[[1, 3, 6]],
             cell="GRUCell", return_sequences=True,
             return_state=False, go_backwards=False,
             stateful=False, unroll=False, time_major=False),
    ]

    @pytest.mark.parametrize("params", test_data_multiple_outputs)
    @pytest.mark.nightly
    def test_keras_rnn_multiple_outputs(self, params, ie_device, precision, ir_version, temp_dir,
                                        use_legacy_frontend):
        self._test(*self.create_keras_rnn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    # Tests for other attributes: go_backward and time_major
    test_data_others = [
        dict(input_names=["x1"], input_shapes=[[2, 4, 5]],
             cell="LSTMCell", return_sequences=False,
             return_state=False, go_backwards=True,
             stateful=False, unroll=False, time_major=False,
             ),
        dict(input_names=["x1"], input_shapes=[[3, 2, 1]],
             cell="SimpleRNNCell", return_sequences=False,
             return_state=False, go_backwards=False,
             stateful=False, unroll=False, time_major=True,
             ),
    ]

    @pytest.mark.parametrize("params", test_data_others)
    @pytest.mark.nightly
    def test_keras_rnn_others(self, params, ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        self._test(*self.create_keras_rnn_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
