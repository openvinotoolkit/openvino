# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import tensorflow as tf

tf.compat.v1.reset_default_graph()

# Create the graph and model
with tf.compat.v1.Session() as sess:
    input1 = tf.compat.v1.placeholder(tf.float32, [1, 3, 3, 1], 'inputX1')
    input2 = tf.compat.v1.placeholder(tf.float32, [1, 3, 3, 1], 'inputX2')

    kernel1 = tf.constant(np.random.randn(1, 1, 1, 1), dtype=tf.float32)
    kernel2 = tf.constant(np.random.randn(1, 1, 1, 1), dtype=tf.float32)

    conv2d1 = tf.nn.conv2d(input1, kernel1, strides=[1, 1], padding='VALID')
    conv2d2 = tf.nn.conv2d(input2, kernel2, strides=[1, 1], padding='VALID')

    add1 = tf.add(conv2d1, conv2d2, name="add1")

    relu2a = tf.nn.relu(add1, name="relu2a")
    relu2b = tf.nn.relu(add1, name="relu2b")

    add2 = tf.add(relu2a, relu2b, name="add2")

    tf.nn.relu(add2, name="relu3a")
    tf.nn.relu(add2, name="relu3b")

    tf.compat.v1.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "2in_2out"), '2in_2out.pb', False)
tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "2in_2out"), '2in_2out.pb.frozen', False)
tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "2in_2out"), '2in_2out.pb.frozen_text', True)
