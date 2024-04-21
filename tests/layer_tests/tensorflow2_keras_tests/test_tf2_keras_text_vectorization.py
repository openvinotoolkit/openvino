# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

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
        inputs_data['text_input'] = rng.choice(self.strings_dictionary, input_shape)
        return inputs_data

    def create_text_vectorization_net(self, input_shapes, vocabulary, output_mode, output_sequence_length,
                                      strings_dictionary):
        assert len(input_shapes) > 0
        self.strings_dictionary = strings_dictionary
        tf.keras.backend.clear_session()
        text_input = tf.keras.Input(shape=input_shapes[0][1:], name='text_input',
                                    dtype=tf.string)
        vectorized_text = TextVectorization(vocabulary=vocabulary,
                                            output_mode=output_mode,
                                            output_sequence_length=output_sequence_length,
                                            name='text_vectorizer')(text_input)
        tf2_net = tf.keras.Model(inputs=[text_input], outputs=[vectorized_text])

        return tf2_net, None

    @pytest.mark.parametrize('input_shapes', [[[1, 1]], [[3, 1]]])
    @pytest.mark.parametrize('strings_dictionary',
                             [['hi OpenVINO here  ', '  hello OpenVINO there', 'hello PyTorch here  ',
                               '  hi TensorFlow here', '  hi JAX here \t'],
                              ['привет ОПЕНВИНО здесь  ', '  привет ОпенВИНО там', 'привет Пайторч здесь  ',
                               '  привет ТензорФлоу здесь', '  привет ДЖАКС там \t'],
                              ['這裡你好 OpenVINO ', '你好 OpenVINO 那裡', '你好這裡 PyTorch ',
                               ' 這裡是 TensorFlow', ' 這裡是 JAX \t']
                              ])
    @pytest.mark.parametrize('vocabulary', [['hello', 'there', 'OpenVINO', 'check', 'привет',
                                             'ОПЕНВИНО', 'здесь', 'там', '你好', '那裡', '檢查']])
    @pytest.mark.parametrize('output_mode', ['int'])
    @pytest.mark.parametrize('output_sequence_length', [32, 64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_text_vectorization(self, input_shapes, vocabulary, output_mode, output_sequence_length, strings_dictionary,
                                ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if platform.system() in ('Darwin') or platform.machine() in ['arm', 'armv7l',
                                                                     'aarch64',
                                                                     'arm64',
                                                                     'ARM64']:
            pytest.xfail(reason='126314, 132699: Build tokenizers for ARM and MacOS')
        params = {}
        params['input_shapes'] = input_shapes
        params['vocabulary'] = vocabulary
        params['output_mode'] = output_mode
        params['output_sequence_length'] = output_sequence_length
        params['strings_dictionary'] = strings_dictionary
        self._test(*self.create_text_vectorization_net(**params), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend, **params)
