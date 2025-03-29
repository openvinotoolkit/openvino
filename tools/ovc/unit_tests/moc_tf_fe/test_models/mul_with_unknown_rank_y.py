# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [3], 'x')
    keep_prob = tf.placeholder(tf.float32, None, 'y')
    tf.multiply(x, keep_prob)

    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'mul_with_unknown_rank_y.pbtxt', True)
