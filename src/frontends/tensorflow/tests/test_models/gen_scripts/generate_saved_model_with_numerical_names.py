# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    tf_x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='0')
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='1')
    tf_z = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='2')
    add = tf.add(tf_x, tf_y, name="3")
    sub = tf.subtract(add, tf_z, name="4")
    tf.compat.v1.saved_model.simple_save(sess, os.path.join(sys.argv[1], "saved_model_with_numerical_names"),
                                         inputs={'0': tf_x, '1': tf_y, '2': tf_z}, outputs={'4': sub})
