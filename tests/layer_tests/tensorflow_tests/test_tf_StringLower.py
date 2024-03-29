# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import run_in_jenkins

rng = np.random.default_rng()


class TestStringLower(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']
        inputs_data = {}
        sample_data = rng.choice(self.strings_dictionary, input_shape)
        inputs_data['input:0'] = sample_data
        return inputs_data

    def create_string_lower_net(self, input_shape, encoding, strings_dictionary):
        self.encoding = encoding
        self.strings_dictionary = strings_dictionary

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.string, input_shape, 'input')
            tf.raw_ops.StringLower(input=input, encoding=encoding)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize("encoding", [None, '', 'utf-8'])
    @pytest.mark.parametrize("input_shape", [[], [2], [3, 4], [1, 3, 2]])
    @pytest.mark.parametrize("strings_dictionary",
                             [['UPPER CASE SENTENCE', 'lower case sentence', ' UppEr LoweR CAse SENtence', ' '],
                              ['Первое Предложение', 'второе    предложение', ' ', ' ТРЕТЬЕ ПРЕДЛОЖЕНИЕ '],
                              ['第一句話在這裡', '第二句話在這裡', '第三句話在這裡']])
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ['arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'],
                       reason='Ticket - 126314, 132699')
    def test_string_lower(self, input_shape, encoding, strings_dictionary, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_string_lower_net(input_shape=input_shape, encoding=encoding,
                                                 strings_dictionary=strings_dictionary),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
