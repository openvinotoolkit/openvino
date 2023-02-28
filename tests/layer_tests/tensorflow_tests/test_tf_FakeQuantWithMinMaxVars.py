# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestFakeQuantWithMinMaxVars(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'inputs' in inputs_info, "Test error: inputs_info must contain `input`"
        inputs_shape = inputs_info['inputs']
        inputs_data = {}
        inputs_data['inputs'] = np.random.randint(-10, 10, inputs_shape).astype(np.float32)
        return inputs_data

    def create_fake_quant_with_min_max_vars_net(self, inputs_shape, min_value, max_value, num_bits, narrow_range,
                                                fake_quant_op):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            inputs = tf.compat.v1.placeholder(tf.float32, inputs_shape, 'inputs')
            min = tf.constant(min_value, dtype=tf.float32)
            max = tf.constant(max_value, dtype=tf.float32)
            fake_quant_op(inputs=inputs, min=min, max=max, num_bits=num_bits,
                          narrow_range=narrow_range)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_basic = [
        # test FakeQuantWithMinMaxVars
        dict(inputs_shape=[2, 6, 4], min_value=-3, max_value=4, num_bits=None, narrow_range=None,
             fake_quant_op=tf.raw_ops.FakeQuantWithMinMaxVars),
        dict(inputs_shape=[3, 2, 1, 5], min_value=-4, max_value=5, num_bits=14, narrow_range=True,
             fake_quant_op=tf.raw_ops.FakeQuantWithMinMaxVars),
        dict(inputs_shape=[3, 2, 4], min_value=2, max_value=4, num_bits=10, narrow_range=False,
             fake_quant_op=tf.raw_ops.FakeQuantWithMinMaxVars),
        dict(inputs_shape=[1, 2, 3], min_value=-6, max_value=-3, num_bits=8, narrow_range=True,
             fake_quant_op=tf.raw_ops.FakeQuantWithMinMaxVars),

        # test FakeQuantWithMinMaxVarsPerChannel
        pytest.param(dict(inputs_shape=[2, 6, 4], min_value=[-4, -3, -5, -8], max_value=[4, 7, 9, 5], num_bits=None,
                          narrow_range=None,
                          fake_quant_op=tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel),
                     marks=pytest.mark.xfail(reason="104822"))

    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_fake_quant_with_min_max_vars_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                                use_new_frontend,
                                                use_old_api):
        self._test(*self.create_fake_quant_with_min_max_vars_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
