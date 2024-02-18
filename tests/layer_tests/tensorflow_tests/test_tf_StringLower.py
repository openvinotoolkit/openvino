# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestStringLower(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        strings_dictionary = ['UPPER CASE SENTENCE', 'lower case sentence', ' UppEr LoweR CAse SENtence', ' ',
                              'Oferta polska', 'Предложение по-РУССки', '汉语句子']
        sample_data = rng.choice(strings_dictionary, input_shape)
        if self.encoding == "" or self.encoding is None:
            inputs_data['input'] = np.char.encode(sample_data, encoding='ascii')
        else:
            inputs_data['input'] = np.char.encode(sample_data, encoding='utf-8')
        return inputs_data

    def create_string_lower_net(self, input_shape, encoding):
        self.encoding = encoding

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
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(reason='132695 - Add support of StringLower')
    def test_string_lower(self, input_shape, encoding, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        self._test(*self.create_string_lower_net(input_shape=input_shape, encoding=encoding),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
