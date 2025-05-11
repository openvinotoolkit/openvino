# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    filter = tf.constant(value=0, shape=[3, 3, 6, 6], dtype=tf.float32)
    input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, 10, 10, None], name='input')
    conv = tf.raw_ops.Conv2D(input=input,
                             filter=filter,
                             strides=[1, 1, 1, 1],
                             padding='SAME')
    tf.compat.v1.saved_model.simple_save(sess, os.path.join(sys.argv[1], "conv_with_dynamic_input_channel"),
                                         inputs={'input': input}, outputs={'conv': conv})
