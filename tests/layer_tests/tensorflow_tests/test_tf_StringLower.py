# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.common_utils import generate_ir_ovc
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
    def test_string_lower(self, input_shape, encoding, strings_dictionary, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_string_lower_net(input_shape=input_shape, encoding=encoding,
                                                 strings_dictionary=strings_dictionary),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestStringLowerOVC:
    def prepare_data(self):
        inputs_data = {}
        inputs_data['input:0'] = np.array(['Some sentence', 'ANOTHER sentenCE'], dtype=str)
        ref_data = np.array(['some sentence', 'another sentence'], dtype=str)
        return inputs_data, ref_data

    def create_string_lower_model(self, output_dir):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.string, [2], 'input')
            tf.raw_ops.StringLower(input=input, name='StringLower')
            tf.compat.v1.global_variables_initializer()
            tf.compat.v1.io.write_graph(sess.graph, output_dir, 'model_string_lower.pb', as_text=False)
        return os.path.join(output_dir, 'model_string_lower.pb')

    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_string_lower_with_ovc(self, ie_device, temp_dir, precision):
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        input_model_path = self.create_string_lower_model(temp_dir)
        output_model_path = os.path.join(temp_dir, 'model_string_lower.xml')
        return_code, _, _ = generate_ir_ovc(input_model_path, {'output_model': output_model_path})
        assert return_code == 0, "OVC tool is failed for conversion model {}".format(input_model_path)

        import openvino as ov
        core = ov.Core()
        compiled_model = core.compile_model(output_model_path, ie_device)
        input_data, ref_data = self.prepare_data()
        ov_result = compiled_model(input_data)['StringLower:0']

        assert np.array_equal(ov_result, ref_data), 'OpenVINO result does not match the reference:' \
                                                    'OpenVINO result - {},' \
                                                    'Reference - {}'.format(ov_result, ref_data)
