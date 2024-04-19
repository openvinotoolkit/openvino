# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest

import numpy as np
import tensorflow as tf

# Testing operation CTCGreedyDecoder
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/CTCGreedyDecoder

class TestCTCGreedyDecoder(CommonTFLayerTest):
    # input_shape - shape a tensor for a decoder
    # merge_repeated - bool, enables/disable merge repeated classes in decoder
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_ctcgreedydecoder_placeholder_const_net(self, input_shape, merge_repeated,
                                            ir_version, use_legacy_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->CTCLoss    =>      Placeholder->Transpose->CTCGreedyDecoder->NotEqual->NonZero->Transpose

        """

        if use_legacy_frontend == False:
            pytest.skip('Legacy path isn\'t supported by CTCGreedyDecoder')

        seq_lens = np.array([input_shape[2]], dtype=np.int32)

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_inputs = tf.compat.v1.placeholder(tf.float32, input_shape, "inputs")

            ctc_gd = tf.raw_ops.CTCGreedyDecoder(inputs = tf_inputs, sequence_length = seq_lens, merge_repeated=merge_repeated)
            tf.identity(ctc_gd[0], name='decoded_indices')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        pytest.param(
            dict(
            input_shape = [6, 1, 4],
            ),
            marks=pytest.mark.precommit),
        dict(
            input_shape = [10, 1, 7],
        ),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("merge_repeated", [False, True])
    @pytest.mark.nightly
    def test_ctcgreedydecoder_placeholder_const(self, params, merge_repeated, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.xfail('104860')
        self._test(*self.create_ctcgreedydecoder_placeholder_const_net(**params, ir_version=ir_version,
                                                             use_legacy_frontend=use_legacy_frontend, merge_repeated=merge_repeated),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, merge_repeated=merge_repeated)
