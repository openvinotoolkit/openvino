# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestTFGRUBlockCell(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.zeros(inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_tf_gru_block_cell(self, batch_size, input_size, hidden_size):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = [batch_size, input_size]
            tf_h_prev_shape = [batch_size, hidden_size]
            tf_w_ru_shape = [input_size + hidden_size, hidden_size * 2]
            tr_w_c_shape = [input_size + hidden_size, hidden_size]
            tf_b_ru_shape = [hidden_size * 2]
            tf_b_c_shape = [hidden_size]

            random_generator = np.random.default_rng(2022)

            x = random_generator.uniform(0, 1, tf_x_shape).astype(np.float32)
            h_prev = random_generator.uniform(0, 1, tf_h_prev_shape).astype(np.float32)

            w_ru = random_generator.uniform(0, 1, tf_w_ru_shape).astype(np.float32)
            w_c = random_generator.uniform(0, 1, tr_w_c_shape).astype(np.float32)

            b_ru = random_generator.uniform(0, 1, tf_b_ru_shape).astype(np.float32)
            b_c = random_generator.uniform(0, 1, tf_b_c_shape).astype(np.float32)

            r, u, c, h = tf.raw_ops.GRUBlockCell(x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c, b_ru=b_ru, b_c=b_c, name="TFGRUBlockCell")

            # Dummy Add layer to prevent from Const network, and compare only "h" output
            input_zero = tf.compat.v1.placeholder(tf.as_dtype(np.float32), tf_h_prev_shape, 'Input')
            add = tf.add(h, input_zero)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data = [
        dict(batch_size=1, input_size=6, hidden_size=6),
        dict(batch_size=1, input_size=10, hidden_size=15),
        dict(batch_size=1, input_size=15, hidden_size=10),
        dict(batch_size=2, input_size=6, hidden_size=6),
        dict(batch_size=2, input_size=12, hidden_size=6),
        pytest.param(dict(batch_size=2, input_size=6, hidden_size=12), marks=pytest.mark.precommit_tf_fe),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tf_gru_block_cell(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend):
        if ie_device == 'GPU':
            pytest.skip("Skip TF GRUBlockCell test on GPU")
        self._test(*self.create_tf_gru_block_cell(**params),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, custom_eps=1e-3, **params)
