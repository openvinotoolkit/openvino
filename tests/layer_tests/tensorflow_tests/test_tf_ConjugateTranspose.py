# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation ConjugateTranspose
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ConjugateTranspose


class TestComplexConjugateTranspose(CommonTFLayerTest):

    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'real_part:0' in inputs_info
        real_part_shape = inputs_info['real_part:0']
        assert 'imag_part:0' in inputs_info
        imag_part_shape = inputs_info['imag_part:0']

        inputs_data = {}
        inputs_data['real_part:0'] = 4 * rng.random(real_part_shape).astype(np.float32) - 2
        inputs_data['imag_part:0'] = 4 * rng.random(imag_part_shape).astype(np.float32) - 2

        return inputs_data

    def create_complex_conjugate_transpose_net(self, input_shape, perm):
        """
            TensorFlow net                                 IR net

            Placeholder->ConjugateTranspose    =>       Placeholder->Transpose->Conjugate->Transpose
        """

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            real_part = tf.compat.v1.placeholder(np.float32, input_shape, 'real_part')
            imag_part = tf.compat.v1.placeholder(np.float32, input_shape, 'imag_part')

            complex_input = tf.raw_ops.Complex(real=real_part, imag=imag_part)

            conj_tranpose = tf.raw_ops.ConjugateTranspose(x=complex_input, perm=perm, name="Operation")
            tf.raw_ops.Real(input=conj_tranpose)
            tf.raw_ops.Imag(input=conj_tranpose)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        (dict(input_shape=[1, 2], perm=[1, 0])),
        (dict(input_shape=[1, 2, 3], perm=[2, 1, 0])),
        (dict(input_shape=[1, 2, 3, 4], perm=[0, 3, 2, 1])),
        (dict(input_shape=[1, 2, 3, 4, 5], perm=[0, 2, 1, 3, 4])),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_conjugate_transpose(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        self._test(*self.create_complex_conjugate_transpose_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestConjugateTranspose(CommonTFLayerTest):

    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']

        inputs_data = {}
        inputs_data['input:0'] = np.random.default_rng().random(input_shape).astype(np.float32)

        return inputs_data

    def create_conjugate_transpose_net(self, input_shape, perm):
        """
            TensorFlow net                                 IR net

            Placeholder->ConjugateTranspose    =>       Placeholder->Transpose->Conjugate->Transpose
        """

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(np.float32, input_shape, 'input')

            tf.raw_ops.ConjugateTranspose(x=input, perm=perm, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        (dict(input_shape=[1, 2], perm=[1, 0])),
        (dict(input_shape=[1, 2, 3], perm=[2, 1, 0])),
        (dict(input_shape=[1, 2, 3, 4], perm=[0, 3, 2, 1])),
        (dict(input_shape=[1, 2, 3, 4, 5], perm=[0, 2, 1, 3, 4])),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_conjugate_transpose(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        self._test(*self.create_conjugate_transpose_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
