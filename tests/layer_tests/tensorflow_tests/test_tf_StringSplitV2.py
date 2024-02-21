# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestStringSplitV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        strings_dictionary = ['UPPER<>CASE SENTENCE<>', 'lower case\n\s sentence', ' UppEr LoweR CAse SENtence \t\n',
                              ' ', 'Oferta polska', 'Предложение<> по-РУССки', '<>汉语句子   ']
        inputs_data['input'] = rng.choice(strings_dictionary, input_shape)
        return inputs_data

    def create_string_split_v2_net(self, input_shape, sep, maxsplit):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.string, input_shape, 'input')
            tf.raw_ops.StringSplitV2(input=input, sep=sep, maxsplit=maxsplit)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('input_shape', [[1], [2], [5]])
    @pytest.mark.parametrize('sep', ['', '<>', '\n\s'])
    @pytest.mark.parametrize('maxsplit', [None, -1, 2, 3])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(reason='132671 - Add support of StringSplitV2')
    def test_string_split_v2(self, input_shape, sep, maxsplit,
                             ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        self._test(*self.create_string_split_v2_net(input_shape=input_shape, sep=sep, maxsplit=maxsplit),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
