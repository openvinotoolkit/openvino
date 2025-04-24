# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
with tf.Session() as sess:
    x = tf.placeholder(tf.int32, [None, 3], 'x')
    y = tf.placeholder_with_default(tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32),
                                    [None, 3], 'y')
    tf.add(x, y)

    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'placeholder_with_default.pbtxt', True)
