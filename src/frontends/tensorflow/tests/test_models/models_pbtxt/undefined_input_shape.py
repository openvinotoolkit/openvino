# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import tensorflow.compat.v1 as tf

tf.reset_default_graph()

with tf.Session() as sess:
    x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='y')
    z = tf.placeholder(dtype=tf.float32, shape=None, name='z')
    add = tf.add(x, y, name="add")
    tf.multiply(add, z)

    tf.global_variables_initializer()
    tf.io.write_graph(sess.graph, '.', 'undefined_input_shape.pbtxt', as_text=True)
