# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    x_value = [[1.,2.,3.],[3.,2.,1.]]
    tf_x = tf.Variable(x_value)
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='y')
    tf_z = tf.add(tf_x, tf_y, name="AddOperation")
    sess.run(tf.compat.v1.global_variables_initializer())
    saver = tf.compat.v1.train.Saver([tf_x])
    os.makedirs(os.path.join(sys.argv[1], "metagraph_variables"))
    saver.save(sess, os.path.join(sys.argv[1], "metagraph_variables", "graph"))
