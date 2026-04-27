# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestReverseSequence(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        assert 'seq_lengths:0' in inputs_info
        input_shape = inputs_info['input:0']
        seq_lengths_shape = inputs_info['seq_lengths:0']
        inputs_data = {}
        inputs_data['input:0'] = np.random.randint(-50, 50, input_shape).astype(self.input_type)
        inputs_data['seq_lengths:0'] = np.random.randint(0, self.max_seq_length + 1, seq_lengths_shape).astype(
            self.seq_lengths_type)
        return inputs_data

    def create_reverse_sequence_net(self, input_shape, input_type, seq_lengths_type, seq_dim, batch_dim):
        self.input_type = input_type
        self.seq_lengths_type = seq_lengths_type
        assert 0 <= batch_dim and batch_dim < len(input_shape), "Incorrect `batch_dim` in the test case"
        assert 0 <= seq_dim and seq_dim < len(input_shape), "Incorrect `seq_dim` in the test case"
        self.max_seq_length = input_shape[seq_dim]
        batch_size = input_shape[batch_dim]
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            seq_lengths = tf.compat.v1.placeholder(seq_lengths_type, [batch_size], 'seq_lengths')
            tf.raw_ops.ReverseSequence(input=input, seq_lengths=seq_lengths, seq_dim=seq_dim, batch_dim=batch_dim)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 3], input_type=np.int32, seq_lengths_type=np.int64, seq_dim=1, batch_dim=0),
        dict(input_shape=[3, 6, 4], input_type=np.float32, seq_lengths_type=np.int32, seq_dim=2, batch_dim=0),
        dict(input_shape=[6, 3, 4, 2], input_type=np.float32, seq_lengths_type=np.int32, seq_dim=0, batch_dim=3),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_reverse_sequence_basic(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reverse_sequence_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)


class TestComplexReverseSequence(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        assert 'seq_lengths:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        seq_lengths_shape = inputs_info['seq_lengths:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        inputs_data['seq_lengths:0'] = np.random.randint(
            0, self.max_seq_length + 1, seq_lengths_shape).astype(np.int32)
        return inputs_data

    def create_complex_reverse_sequence_net(self, input_shape, seq_dim, batch_dim):
        assert 0 <= batch_dim < len(input_shape), "Incorrect batch_dim in the test case"
        assert 0 <= seq_dim < len(input_shape), "Incorrect seq_dim in the test case"
        self.max_seq_length = input_shape[seq_dim]
        batch_size = input_shape[batch_dim]
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            seq_lengths = tf.compat.v1.placeholder(np.int32, [batch_size], 'seq_lengths')
            complex_tensor = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            reverse_sequence = tf.raw_ops.ReverseSequence(
                input=complex_tensor,
                seq_lengths=seq_lengths,
                seq_dim=seq_dim,
                batch_dim=batch_dim)
            tf.raw_ops.Real(input=reverse_sequence)
            tf.raw_ops.Imag(input=reverse_sequence)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        return tf_net, None

    test_data_complex = [
        dict(input_shape=[2, 3], seq_dim=1, batch_dim=0),
        dict(input_shape=[3, 6, 4], seq_dim=2, batch_dim=0),
        dict(input_shape=[6, 3, 4, 2], seq_dim=0, batch_dim=3),
    ]

    @pytest.mark.parametrize("params", test_data_complex)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_reverse_sequence(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(
            *self.create_complex_reverse_sequence_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir)
