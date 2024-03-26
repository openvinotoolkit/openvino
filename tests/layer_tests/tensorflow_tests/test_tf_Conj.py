# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation Conj
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Conj

class TestComplexConjugate(CommonTFLayerTest):

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

            conj = tf.raw_ops.Conj(input=complex_input, name="Operation")
            tf.raw_ops.Real(input=conj)
            tf.raw_ops.Imag(input=conj)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize("input_shape", [[1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_conjugate(self, input_shape, ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        self._test(*self.create_complex_conjugate_net(input_shape),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
