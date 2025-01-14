# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(2024)


class TestGRUBlockCell(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(0, 1, x_shape).astype(np.float32)
        return inputs_data

    def create_gru_block_cell(self, batch_size, input_size, hidden_size):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x_shape = [batch_size, input_size]
            h_prev_shape = [batch_size, hidden_size]
            w_ru_shape = [input_size + hidden_size, hidden_size * 2]
            w_c_shape = [input_size + hidden_size, hidden_size]
            b_ru_shape = [hidden_size * 2]
            b_c_shape = [hidden_size]

            x = tf.compat.v1.placeholder(np.float32, x_shape, 'x')

            # generate weights
            h_prev = rng.uniform(0, 1, h_prev_shape).astype(np.float32)
            w_ru = rng.uniform(0, 1, w_ru_shape).astype(np.float32)
            w_c = rng.uniform(0, 1, w_c_shape).astype(np.float32)
            b_ru = rng.uniform(0, 1, b_ru_shape).astype(np.float32)
            b_c = rng.uniform(0, 1, b_c_shape).astype(np.float32)

            _, _, _, h = tf.raw_ops.GRUBlockCell(x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c, b_ru=b_ru, b_c=b_c)
            tf.identity(h, name='gru_block_cell')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    @pytest.mark.parametrize('batch_size', [1, 2])
    @pytest.mark.parametrize('input_size', [1, 5, 6, 12])
    @pytest.mark.parametrize('hidden_size', [1, 6, 10, 12])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_gru_block_cell(self, batch_size, input_size, hidden_size,
                            ie_device, precision, ir_version, temp_dir,
                            use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Skip TF GRUBlockCell test on GPU")
        self._test(*self.create_gru_block_cell(batch_size, input_size, hidden_size),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=1e-3)
