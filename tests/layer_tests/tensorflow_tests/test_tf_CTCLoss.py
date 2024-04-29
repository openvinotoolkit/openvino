# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation CTCLoss
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/CTCLoss

class TestCTCLoss(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(0, 5, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_ctcloss_placeholder_const_net(self, inputs, targets, preprocess_collapse_repeated, ctc_merge_repeated):
        seq_lens = np.array([inputs[2]], dtype=np.int32)
        x = [targets]

        indices = []
        vals = []
        for idx, batch in enumerate(x):
            for time, value in enumerate(batch):
                indices.append([idx, time])
                vals.append(value)

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_inputs = tf.compat.v1.placeholder(tf.float32, inputs, "inputs")

            ctc_loss = tf.raw_ops.CTCLoss(inputs=tf_inputs, labels_indices=indices, labels_values=vals,
                                          sequence_length=seq_lens,
                                          preprocess_collapse_repeated=preprocess_collapse_repeated,
                                          ctc_merge_repeated=ctc_merge_repeated)
            # compute exponent since CTCLoss value is -ln(prob)
            tf.math.exp(-ctc_loss[0])

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # Reference values were copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/nn_ops/ctc_loss_op_test.py
    test_data = [
        dict(inputs=[6, 1, 6], targets=[0, 1, 2, 1, 0]),
        dict(inputs=[12, 1, 9], targets=[0, 1, 1, 0])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("preprocess_collapse_repeated", [True, False, None])
    @pytest.mark.parametrize("ctc_merge_repeated", [True, False, None])
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_ctcloss_placeholder_const(self, params, preprocess_collapse_repeated, ctc_merge_repeated,
                                       ie_device, precision, ir_version, temp_dir,
                                       use_legacy_frontend):
        self._test(*self.create_ctcloss_placeholder_const_net(**params,
                                                              preprocess_collapse_repeated=preprocess_collapse_repeated,
                                                              ctc_merge_repeated=ctc_merge_repeated),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
