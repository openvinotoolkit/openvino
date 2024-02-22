# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest
from tensorflow.keras.layers import TextVectorization

rng = np.random.default_rng()


class TestTextVectorization(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        assert 'text_input' in inputs_info
        input_shape = inputs_info['text_input']
        inputs_data = {}
        strings_dictionary = ['hi OpenVINO here  ', '  hello OpenVINO there', ' привет ОПЕНВИНО  \n',
                              'hello PyTorch here  ', '  hi TensorFlow here', '  hi JAX here \t']
        inputs_data['text_input'] = rng.choice(strings_dictionary, input_shape)
        return inputs_data

    def create_text_vectorization_net(self, input_shape, vocabulary, output_mode, output_sequence_length):
        assert len(input_shape) > 0
        tf.keras.backend.clear_session()
        text_input = tf.keras.Input(shape=input_shape[1:], name='text_input',
                                    dtype=tf.string)
        vectorized_text = TextVectorization(vocabulary=vocabulary,
                                            output_mode=output_mode,
                                            output_sequence_length=output_sequence_length,
                                            name='text_vectorizer')(text_input)
        tf2_net = tf.keras.Model(inputs=[text_input], outputs=[vectorized_text])

        return tf2_net, None

    @pytest.mark.parametrize('input_shape', [[2, 1], [2, 3]])
    @pytest.mark.parametrize('vocabulary', [['hello', 'there', 'OpenVINO', 'check', 'привет', 'ОПЕНВИНО']])
    @pytest.mark.parametrize('output_mode', ['int'])
    @pytest.mark.parametrize('output_sequence_length', [32, 64])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.xfail(reason='132692 - Add support of TextVectorization')
    @pytest.mark.nightly
    def test_text_vectorization(self, input_shape, vocabulary, output_mode, output_sequence_length, ie_device,
                                precision, ir_version, temp_dir, use_legacy_frontend):
        params = {}
        params['input_shape'] = input_shape
        params['vocabulary'] = vocabulary
        params['output_mode'] = output_mode
        params['output_sequence_length'] = output_sequence_length
        self._test(*self.create_text_vectorization_net(**params), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend, **params)
