# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
# Create the graph and model
with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], 'x')
    y = tf.placeholder(tf.float32, [2, 3], 'y')
    cond = tf.constant(True, dtype=tf.bool)
    message1 = tf.constant("TensorFlow Frontend", dtype=tf.string)
    message2 = tf.constant("TensorFlow Frontend, ONNX Frontend", dtype=tf.string)
    message3 = tf.constant("TensorFlow Frontend, ONNX Frontend, PDPD Frontend", dtype=tf.string)
    select = tf.where(cond, x, y)
    assert_op = tf.Assert(cond, [message1, message2, message3])

    tf.global_variables_initializer()
    tf.io.write_graph(sess.graph, '.', 'string_tensors_model.pbtxt', as_text=True)
