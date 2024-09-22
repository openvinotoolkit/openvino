# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

# Testing operation ArgMin, ArgMax (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMin
#                https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMax

rng = np.random.default_rng(2323534)


class TestArgMinMax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info, "Test error: inputs_info must contain `input`"
        x_shape = inputs_info['input:0']
        inputs_data = {}
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['input:0'] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data['input:0'] = rng.integers(-8, 8, x_shape).astype(self.input_type)
        else:
            inputs_data['input:0'] = rng.integers(0, 8, x_shape).astype(self.input_type)
        return inputs_data

    def create_argmin_max_net(self, input_shape, dimension, input_type, output_type, op_type):
        OPS = {
            'tf.raw_ops.ArgMax': tf.raw_ops.ArgMax,
            'tf.raw_ops.ArgMin': tf.raw_ops.ArgMin
        }

        self.input_type = input_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            OPS[op_type](input=input, dimension=dimension, output_type=output_type)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('input_shape, dimension', [([10], 0),
                                                        ([10, 15], 1),
                                                        ([10, 15, 20], 2),
                                                        ([10, 15, 20], -1)
                                                        ])
    @pytest.mark.parametrize('input_type', [np.float16, np.float32, np.float64,
                                            np.int32, np.uint8, np.int16, np.int8, np.int64,
                                            np.uint16, np.uint32, np.uint64])
    @pytest.mark.parametrize('op_type, output_type', [('tf.raw_ops.ArgMax', np.int16),
                                                      ('tf.raw_ops.ArgMax', np.uint16),
                                                      ('tf.raw_ops.ArgMax', np.int32),
                                                      ('tf.raw_ops.ArgMax', np.int64),
                                                      # tf.raw_ops.ArgMin supports only int32 and int64
                                                      ('tf.raw_ops.ArgMin', np.int32),
                                                      ('tf.raw_ops.ArgMin', np.int64)
                                                      ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_argmin_max_net(self, input_shape, dimension, input_type, output_type, op_type,
                            ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if platform.machine() in ['aarch64', 'arm64', 'ARM64']:
            pytest.skip('153077: Segmentation fault on ARM')
        if ie_device == 'GPU' and input_type == np.uint8:
            pytest.skip('153078: No layout format available for topk')
        if ie_device == 'GPU' and input_type == np.float32 and input_shape == [10, 15, 20]:
            pytest.skip('153079: Accuracy error on GPU')
        self._test(*self.create_argmin_max_net(input_shape=input_shape, dimension=dimension,
                                               input_type=input_type, output_type=output_type,
                                               op_type=op_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
