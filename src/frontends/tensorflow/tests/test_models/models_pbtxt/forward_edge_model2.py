# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import tensorflow.compat.v1 as tf

tf.reset_default_graph()

with tf.Session() as sess:
    const2 = tf.constant(2.0, dtype=tf.float32)
    x = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='x')
    relu = tf.nn.relu(x)    
    mul = tf.multiply(relu, const2, name="mul")
    # it has forward-edge from relu to multiply
    # i.e. edge skipping direct child
    tf.add(relu, mul)

    tf.global_variables_initializer()
    tf.io.write_graph(sess.graph, '.', 'forward_edge_model2.pb', as_text=False)
