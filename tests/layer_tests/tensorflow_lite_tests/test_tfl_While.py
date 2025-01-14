# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf

from common.tflite_layer_test_class import TFLiteLayerTest


class TestTFLiteWhileLayerTest(TFLiteLayerTest):
    inputs = ["Input"]
    outputs = ["While"]
    allowed_ops = ["ADD", "LESS", "RESHAPE", "WHILE"]

    def make_model(self, params):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            i = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name=self.inputs[0])
            c = lambda i: tf.less(i, [10])
            b = lambda i: (tf.add(i, [1]), )
            tf.while_loop(c, b, [i], name=self.outputs[0])
            net = sess.graph_def
        return net

    @pytest.mark.parametrize("params", [dict()])
    @pytest.mark.nightly
    def test_while(self, params, ie_device, precision, temp_dir):
        pytest.xfail("CVS-112884")
        self._test(ie_device, precision, temp_dir, params)
