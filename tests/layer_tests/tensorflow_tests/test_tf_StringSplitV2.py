# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import run_in_jenkins

rng = np.random.default_rng()


class TestStringSplitV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']
        inputs_data = {}
        strings_dictionary = ['UPPER<>CASE SENTENCE', 'lower case\n sentence', ' UppEr LoweR CAse SENtence \t\n',
                              '  some sentence', 'another sentence HERE    ']
        inputs_data['input:0'] = rng.choice(strings_dictionary, input_shape)
        return inputs_data

    def create_string_split_v2_net(self, input_shape, sep, maxsplit):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.string, input_shape, 'input')
            string_split_v2 = tf.raw_ops.StringSplitV2(input=input, sep=sep, maxsplit=maxsplit)
            tf.identity(string_split_v2[0], name='indices')
            tf.identity(string_split_v2[1], name='values')
            tf.identity(string_split_v2[2], name='shape')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('input_shape', [[1], [2], [5]])
    @pytest.mark.parametrize('sep', ['', '<>'])
    @pytest.mark.parametrize('maxsplit', [None, -1, 5, 10])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_string_split_v2(self, input_shape, sep, maxsplit,
                             ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_string_split_v2_net(input_shape=input_shape, sep=sep, maxsplit=maxsplit),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
