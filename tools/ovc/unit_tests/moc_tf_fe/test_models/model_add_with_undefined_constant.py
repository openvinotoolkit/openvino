# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
# Create the graph and model
with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], 'x')
    const = tf.constant(value=[], dtype=tf.float32, shape=[3], name='Const')
    tf.add(x, const, name="add")
    tf.global_variables_initializer()
    tf.io.write_graph(sess.graph, './', 'model_add_with_undefined_constant.pbtxt', as_text=True)
