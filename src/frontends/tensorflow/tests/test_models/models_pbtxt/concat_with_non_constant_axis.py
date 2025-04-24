# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import tensorflow.compat.v1 as tf

tf.reset_default_graph()

with tf.Session() as sess:
    x = tf.placeholder(dtype=tf.float32, shape=[4, 3], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='y')
    z = tf.placeholder(dtype=tf.float32, shape=[1, 3], name='z')
    const1 = tf.constant(-1, dtype=tf.int32)
    const2 = tf.constant(1, dtype=tf.int32)
    axis = tf.add(const1, const2, name="axis")
    tf.concat([x, y, z], axis)

    tf.global_variables_initializer()
    tf.io.write_graph(sess.graph, '.', 'concat_with_non_constant_axis.pbtxt', as_text=True)
