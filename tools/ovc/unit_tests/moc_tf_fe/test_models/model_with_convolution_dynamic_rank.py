# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
with tf.Session() as sess:
    x = tf.placeholder(tf.float32, None, 'x')
    filter = tf.placeholder(tf.float32, [2, 2, 3, 1], 'kernel')

    conv2d = tf.raw_ops.Conv2D(input=x, filter=filter, strides=[1, 1, 1, 1], padding='SAME',
                               dilations=None)
    relu = tf.raw_ops.Relu(features=conv2d)

    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'model_with_convolution_dynamic_rank.pbtxt', True)
