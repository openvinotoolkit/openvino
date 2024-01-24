# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

# Testing operation Conj
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Conj

class TestComplexConjugate(CommonTFLayerTest):

    def _prepare_input(self, inputs_info):

        rng = np.random.default_rng()
        assert 'real_part' in inputs_info
        real_part_shape = inputs_info['real_part']
        assert 'imag_part' in inputs_info
        imag_part_shape = inputs_info['imag_part']

        inputs_data = {}
        inputs_data['real_part'] = 4 * rng.random(real_part_shape).astype(np.float32) - 2
        inputs_data['imag_part'] = 4 * rng.random(imag_part_shape).astype(np.float32) - 2

        return inputs_data
    def create_complex_conjugate_net(self, input_shape):
        """
            TensorFlow net                                 IR net
            Placeholder->Conjugate    =>       Placeholder->Conjugate
        """

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            real_part = tf.compat.v1.placeholder(np.float32, input_shape, 'real_part')
            imag_part = tf.compat.v1.placeholder(np.float32, input_shape, 'imag_part')

            complex_input = tf.raw_ops.Complex(real=real_part, imag=imag_part)

            conj= tf.raw_ops.Conj(x=complex_input, name = "Operation")
            real = tf.raw_ops.Real(input=conj)
            img = tf.raw_ops.Imag(input=conj)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net


    @pytest.mark.parametrize("params", [[1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5,6]])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_conjugate(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_new_frontend, use_old_api):
        self._test(*self.create_complex_conjugate_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestConjugate(CommonTFLayerTest):

    def _prepare_input(self, inputs_info):

        assert 'input' in inputs_info
        input_shape = inputs_info['input']

        inputs_data = {}
        inputs_data['input'] = np.random.default_rng().random(input_shape).astype(np.float32)

        return inputs_data

    def create_conjugate_net(self, input_shape):
        """
            TensorFlow net                                 IR net
            Placeholder->Conjugate    =>       Placeholder->Conjugate
        """

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(np.float32, input_shape, 'input')

            tf.raw_ops.Conj(x=input, name = "Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize("params", [[1,2], [1,2,3], [1,2,3,4], [1,2,3,4,5,6]])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_conjugate(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_new_frontend, use_old_api):
        self._test(*self.create_conjugate_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api) 