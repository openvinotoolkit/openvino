# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestStaticRegexReplace(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        strings_dictionary = ['UPPER CASE SENTENCE', 'lower case sentence', ' UppEr LoweR CAse SENtence \t\n', ' ',
                              'Oferta polska', 'Предложение по-РУССки', '汉语句子']
        inputs_data['input'] = rng.choice(strings_dictionary, input_shape)
        return inputs_data

    def create_static_regex_replace_net(self, input_shape, pattern, rewrite, replace_global):
        self.pattern = pattern

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.string, input_shape, 'input')
            tf.raw_ops.StaticRegexReplace(input=input, pattern=pattern, rewrite=rewrite, replace_global=replace_global)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('input_shape', [[], [2], [3, 4], [1, 3, 2]])
    @pytest.mark.parametrize('pattern', ['(\s)|(-)', '[A-Z]{2,}', '^\s+|\s+$'])
    @pytest.mark.parametrize('rewrite', ['', 'replacement word'])
    @pytest.mark.parametrize('replace_global', [None, True, False])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(reason='132674 - Add support of StaticRegexReplace')
    def test_static_regex_replace(self, input_shape, pattern, rewrite, replace_global,
                                  ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_static_regex_replace_net(input_shape=input_shape, pattern=pattern, rewrite=rewrite,
                                                         replace_global=replace_global),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
