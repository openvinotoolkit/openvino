# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
with tf.Session() as sess:
    x = tf.placeholder(tf.bool, [2, 3], 'in1')
    y = tf.placeholder(tf.bool, [2, 3], 'in2')
    tf.math.logical_and(x, y)

    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'model_bool.pbtxt', True)
