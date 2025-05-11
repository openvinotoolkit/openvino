# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    input1 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2], name='input1')
    input2 = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2], name='input1')
    output1 = tf.add(input1, input2, name="output1")
    output2 = tf.subtract(output1, input2, name="output2")
    tf.compat.v1.saved_model.simple_save(sess, os.path.join(sys.argv[1], "saved_model_intermediate_output"),
                                         inputs={'input1': input1, 'input2': input2},
                                         outputs={'output1': output1, 'output2': output2})
