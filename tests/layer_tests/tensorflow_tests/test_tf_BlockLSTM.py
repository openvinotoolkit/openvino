# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(2024)


class TestBlockLSTM(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(0, 1, x_shape).astype(np.float32)
        return inputs_data

    def create_block_lstm(self, time_len, seq_len_max, input_size, hidden_size, batch_size,
                          forget_bias, cell_clip, use_peephole):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(np.float32, [time_len, batch_size, input_size], 'x')
            cs_prev = rng.uniform(0, 1, [batch_size, hidden_size]).astype(np.float32)
            h_prev = rng.uniform(0, 1, [batch_size, hidden_size]).astype(np.float32)
            w = rng.uniform(0, 1, [input_size + hidden_size, 4 * hidden_size]).astype(np.float32)
            wci = rng.uniform(0, 1, [hidden_size]).astype(np.float32)
            wcf = rng.uniform(0, 1, [hidden_size]).astype(np.float32)
            wco = rng.uniform(0, 1, [hidden_size]).astype(np.float32)
            b = rng.uniform(0, 1, [4 * hidden_size]).astype(np.float32)
            _, cs_output, _, _, _, _, h_output = tf.raw_ops.BlockLSTM(x=x, seq_len_max=seq_len_max, cs_prev=cs_prev,
                                                                      h_prev=h_prev, w=w, wci=wci,
                                                                      wcf=wcf, wco=wco, b=b, forget_bias=forget_bias,
                                                                      cell_clip=cell_clip,
                                                                      use_peephole=use_peephole)
            tf.identity(cs_output, name='cs_output')
            tf.identity(h_output, name='h_output')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    @pytest.mark.parametrize('time_len', [2, 5])
    @pytest.mark.parametrize('seq_len_max', [1, 2])
    @pytest.mark.parametrize('input_size', [1, 2, 5])
    @pytest.mark.parametrize('hidden_size', [1, 3])
    @pytest.mark.parametrize('batch_size', [1, 2, 3])
    @pytest.mark.parametrize('forget_bias', [-2.0, 0.0, 1.0])
    @pytest.mark.parametrize('cell_clip', [-1.0])
    @pytest.mark.parametrize("use_peephole", [False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_block_lstm(self, time_len, seq_len_max, input_size, hidden_size, batch_size,
                        forget_bias, cell_clip, use_peephole,
                        ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Skip BlockLSTM test on GPU")
        self._test(*self.create_block_lstm(time_len, seq_len_max, input_size, hidden_size, batch_size,
                                           forget_bias, cell_clip, use_peephole),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=3 * 1e-3)
