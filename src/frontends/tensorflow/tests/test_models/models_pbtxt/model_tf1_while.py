# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()

# Note: run this script in TensorFlow 1 environment to generate model_tf1_while.pbtxt
# The model with Switch, NextIteration and other TF1 While stuff cannot be generated in TF2 environment
with tf.Session() as sess:
    i = tf.placeholder(tf.int32, [], 'i')
    j = tf.placeholder(tf.int32, [], 'j')

    r = tf.while_loop(lambda i: tf.less(i, 10), lambda i: (tf.add(i, 1),), [i])
    tf.add(r, j)
    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'model_tf1_while.pbtxt', True)
