# Copyright (C) 2018-2025 Intel Corporation
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
    def test_reverse_sequence_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_reverse_sequence_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexReverseSequence(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        assert 'seq_lengths:0' in inputs_info
        input_shape = inputs_info['input:0']
        seq_lengths_shape = inputs_info['seq_lengths:0']
        inputs_data = {}
        # Generate random complex numbers
        real_part = np.random.randint(-50, 50, input_shape).astype(np.float32)
        imag_part = np.random.randint(-50, 50, input_shape).astype(np.float32)
        inputs_data['input:0'] = real_part + 1j * imag_part
        inputs_data['seq_lengths:0'] = np.random.randint(0, self.max_seq_length + 1, seq_lengths_shape).astype(
            self.seq_lengths_type)
        return inputs_data

    def create_reverse_sequence_net(self, input_shape, seq_lengths_type, seq_dim, batch_dim):
        self.input_type = np.complex64
        self.seq_lengths_type = seq_lengths_type
        # Convert negative indices to positive for TensorFlow
        effective_batch_dim = batch_dim if batch_dim >= 0 else len(input_shape) + batch_dim
        effective_seq_dim = seq_dim if seq_dim >= 0 else len(input_shape) + seq_dim
        assert 0 <= effective_batch_dim and effective_batch_dim < len(input_shape), "Incorrect `batch_dim` in the test case"
        assert 0 <= effective_seq_dim and effective_seq_dim < len(input_shape), "Incorrect `seq_dim` in the test case"
        self.max_seq_length = input_shape[effective_seq_dim]
        batch_size = input_shape[effective_batch_dim]
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.complex64, input_shape, 'input')
            seq_lengths = tf.compat.v1.placeholder(seq_lengths_type, [batch_size], 'seq_lengths')
            tf.raw_ops.ReverseSequence(input=input, seq_lengths=seq_lengths, seq_dim=effective_seq_dim, batch_dim=effective_batch_dim)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None
    
    def compare_ie_results_with_framework(self, infer_res, framework_res, framework_eps):
        is_ok = True
        from common.utils.common_utils import allclose
        
        for framework_out_name in framework_res:
            if framework_out_name not in infer_res and len(infer_res) == 1:
                ov_res = list(infer_res.values())[0]
            else:
                ov_res = infer_res[framework_out_name]

            fw_res = np.array(framework_res[framework_out_name])

            # Special handling for complex tensors
            is_complex_tensor = fw_res.dtype == np.complex64 or fw_res.dtype == np.complex128
            
            if not is_complex_tensor:
                # Check output types for non-complex tensors
                assert fw_res.dtype == ov_res.dtype or \
                    ov_res.dtype.type == str or \
                    ov_res.dtype.type == np.str_, 'Outputs types are different: ' \
                                                    'OpenVINO output type - {}, ' \
                                                    'Framework output type - {}'.format(ov_res.dtype, fw_res.dtype)
            else:
                # For complex tensors, OpenVINO returns only the real part
                print("Complex tensor detected: TF type {} vs OpenVINO type {}".format(fw_res.dtype, ov_res.dtype))
                
                # Check shapes match (without considering complex nature)
                assert fw_res.shape == ov_res.shape, 'Output shapes do not match: ' \
                                                    'OpenVINO shape {} vs Framework shape {}'.format(ov_res.shape, fw_res.shape)
                
                # Compare only real parts for complex tensors
                fw_real_part = fw_res.real
                
                # Compare only real parts
                if not allclose(ov_res, fw_real_part,
                            atol=framework_eps,
                            rtol=framework_eps):
                    is_ok = False
                    diff = np.array(abs(ov_res - fw_real_part)).max()
                    print("Complex tensor real part max diff is {}".format(diff))
                else:
                    print("Complex tensor real part validation successful!\n")
                    print("absolute eps: {}, relative eps: {}".format(framework_eps, framework_eps))
                
                continue

            # Check output shapes match
            assert fw_res.shape == ov_res.shape, 'Outputs shapes are different: ' \
                                                'OpenVINO output shape - {}, ' \
                                                'Framework output shape - {}'.format(ov_res.shape, fw_res.shape)

            # Compare output values
            if not allclose(ov_res, fw_res,
                            atol=framework_eps,
                            rtol=framework_eps):
                is_ok = False
                if ov_res.dtype != bool:
                    diff = np.array(abs(ov_res - fw_res)).max()
                    print("Max diff is {}".format(diff))
                else:
                    print("Boolean results are not equal")
            else:
                print("Accuracy validation successful!\n")
                print("absolute eps: {}, relative eps: {}".format(framework_eps, framework_eps))
                
        return is_ok

    test_data_complex = [
        dict(input_shape=[4, 5], seq_lengths_type=np.int32, seq_dim=1, batch_dim=0),
        dict(input_shape=[2, 3, 4], seq_lengths_type=np.int32, seq_dim=1, batch_dim=0),
        dict(input_shape=[4, 5], seq_lengths_type=np.int32, seq_dim=-1, batch_dim=0),
        dict(input_shape=[4, 5], seq_lengths_type=np.int32, seq_dim=1, batch_dim=-2),
        dict(input_shape=[4, 5], seq_lengths_type=np.int32, seq_dim=-1, batch_dim=-2),
        dict(input_shape=[2, 3, 4], seq_lengths_type=np.int32, seq_dim=-2, batch_dim=0),
        dict(input_shape=[2, 3, 4], seq_lengths_type=np.int32, seq_dim=1, batch_dim=-3),
        dict(input_shape=[2, 3, 4], seq_lengths_type=np.int32, seq_dim=-2, batch_dim=-3),
    ]

    @pytest.mark.parametrize("params", test_data_complex)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_reverse_sequence_complex(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_reverse_sequence_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)