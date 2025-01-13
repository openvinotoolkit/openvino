# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import tensorflow.compat.v1 as tf

tf.reset_default_graph()

with tf.Session() as sess:
    x = tf.placeholder(dtype=tf.float32, shape=[2,3], name='x')
    tf.nn.relu(x, name="relu")

    tf.global_variables_initializer()
    sess.graph.as_graph_def(add_shapes=True)
    tf.io.write_graph(sess.graph, '.', 'model_with_output_shapes_attr.pbtxt', as_text=True)
