# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation AddN
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/AddN

class TestAddN(CommonTFLayerTest):
    # input_shapes - should be an array, could be a single shape, or array of n-dimentional shapes
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_addn_placeholder_const_net(self, input_shapes, ir_version, use_legacy_frontend):
        """
            Tensorflow net                  IR net

            Placeholder_1->AddN    =>       Placeholder_1->AddN
            ...           /                 ...           /
            Placeholder_N/                  Placeholder_N/

        """

        if len(input_shapes) == 0:
            raise RuntimeError("Input list couldn't be empty")

        if len(input_shapes) == 1 and use_legacy_frontend:
            pytest.xfail(reason="96687")

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_inputs = []

            for idx, input_shape in enumerate(input_shapes):
                tf_inputs.append(tf.compat.v1.placeholder(tf.float32, input_shape, f"Input_{idx}"))

            tf.raw_ops.AddN(inputs = tf_inputs)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shapes=[[4]]),                             # Tests sum of scalar values in a single shape
        pytest.param(
            dict(input_shapes=[[4, 3], [4, 3]]),              # Tests sum of shapes
            marks=pytest.mark.precommit),
        dict(input_shapes=[[3, 4, 5], [3, 4, 5], [3, 4, 5]]), # Tests sum of shapes which may trigger nchw/nhcw transformation
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_addn_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_addn_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

class TestComplexAddN(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        inputs_data = {}
        for idx, key in enumerate(inputs_info):
            assert key in inputs_info
            inputs_data[key] = 4 * rng.random(inputs_info[key]).astype(np.float32) - 2
        return inputs_data

    def create_complex_addn_net(self, input_shapes):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            complex_tensors = []
            for idx, input_shape in enumerate(input_shapes):
                real = tf.compat.v1.placeholder(np.float32, input_shape, f'param_real_{idx+1}')
                imag = tf.compat.v1.placeholder(np.float32, input_shape, f'param_imag_{idx+1}')
                complex_tensors.append(tf.raw_ops.Complex(real=real, imag=imag))
            addn = tf.raw_ops.AddN(inputs=complex_tensors, name='complex_AddN')
            real = tf.raw_ops.Real(input=addn)
            imag = tf.raw_ops.Imag(input=addn)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        return tf_net, None

    test_data = [
        dict(input_shapes=[[1], [1]]),
        dict(input_shapes=[[2, 3], [2, 3], [2, 3], [2, 3]]),
        dict(input_shapes=[[3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5], [3, 4, 5]]),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_addn(self, params, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        self._test(*self.create_complex_addn_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
