# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf

tf.reset_default_graph()
with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [2, 3], 'x')
    y = tf.placeholder(tf.float32, [2, 3], 'y')
    is_training = tf.placeholder(tf.bool, [], 'is_training')
    switch = tf.raw_ops.Switch(data=x, pred=is_training)
    relu = tf.raw_ops.Relu(features=switch[0])
    sigmoid = tf.raw_ops.Sigmoid(x=switch[1])
    merge = tf.raw_ops.Merge(inputs=[relu, sigmoid])
    tf.raw_ops.AddV2(x=merge[0], y=y)

    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'model_switch_merge.pbtxt', True)
