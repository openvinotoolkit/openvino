# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest

import numpy as np
import tensorflow as tf

# Testing operation CTCLoss
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/CTCLoss

class TestCTCLoss(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(0, 5, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_ctcloss_placeholder_const_net(self, inputs, targets, ir_version, use_new_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->CTCLoss    =>      Placeholder->Transpose->Convolution->Transpose #Need to replace by actual

        """
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

            tf.raw_ops.CTCLoss(inputs = tf_inputs, labels_indices = indices, labels_values = vals, sequence_length = seq_lens)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # Reference values were copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/kernel_tests/nn_ops/ctc_loss_op_test.py
    test_data = [
        dict(inputs=[6,1,6], targets = [0, 1, 2, 1, 0]),
        dict(inputs=[12,1,9], targets = [0, 1, 1, 0])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_ctcloss_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_ctcloss_placeholder_const_net(**params, ir_version=ir_version,
                                                             use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
