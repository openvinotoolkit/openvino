# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


OPS = {
    'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel': tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel,
    'tf.raw_ops.FakeQuantWithMinMaxVars':  tf.raw_ops.FakeQuantWithMinMaxVars,
    'tf.raw_ops.FakeQuantWithMinMaxArgs': tf.raw_ops.FakeQuantWithMinMaxArgs
}

class TestFakeQuantWithMinMaxVars(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'inputs:0' in inputs_info, "Test error: inputs_info must contain `input`"
        inputs_shape = inputs_info['inputs:0']
        inputs_data = {}
        inputs_data['inputs:0'] = np.random.randint(-10, 10, inputs_shape).astype(np.float32)
        return inputs_data

    def create_fake_quant_with_min_max_vars_net(self, inputs_shape, min_value, max_value, num_bits, narrow_range,
                                                fake_quant_op):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            inputs = tf.compat.v1.placeholder(tf.float32, inputs_shape, 'inputs')
            fake_quant_op(inputs=inputs, min=min_value, max=max_value, num_bits=num_bits,
                          narrow_range=narrow_range)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_basic = [
        # test FakeQuantWithMinMaxVars
        [[2, 6, 4], -3, 4, None, None],
        [[3, 2, 1, 5], -4, 5, 14, True],
        [[3, 2, 4], 2, 4, 10, False],
        [[1, 2, 3], -6, -3, 8, True],
    ]

    @pytest.mark.parametrize("inputs_shape, min_value, max_value, num_bits, narrow_range", test_basic)
    @pytest.mark.parametrize("fake_quant_op", [
        'tf.raw_ops.FakeQuantWithMinMaxVars', 'tf.raw_ops.FakeQuantWithMinMaxArgs'
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_fake_quant_with_min_max_vars_basic(self, inputs_shape, min_value, max_value, num_bits, narrow_range, fake_quant_op, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        params = dict(inputs_shape=inputs_shape, min_value=min_value, max_value=max_value, num_bits=num_bits, narrow_range=narrow_range)
        self._test(*self.create_fake_quant_with_min_max_vars_net(**params, fake_quant_op=OPS[fake_quant_op]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_per_channel_basic = [
        [[2, 6, 4], [-4, -3, -5, -8], [4, 7, 9, 5], None, None, 'tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel'],
    ]

    @pytest.mark.parametrize("inputs_shape, min_value, max_value, num_bits, narrow_range, fake_quant_op", test_per_channel_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail("104822")
    def test_fake_quant_with_min_max_vars_per_channel_basic(self, inputs_shape, min_value, max_value, num_bits, narrow_range, fake_quant_op, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        params=dict(inputs_shape=inputs_shape, min_value=min_value, max_value=max_value, num_bits=num_bits, narrow_range=narrow_range, fake_quant_op=OPS[fake_quant_op])
        self._test(*self.create_fake_quant_with_min_max_vars_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
