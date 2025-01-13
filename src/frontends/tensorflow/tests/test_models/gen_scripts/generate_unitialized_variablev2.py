# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

tf.compat.v1.reset_default_graph()
# Create the graph and model
with tf.compat.v1.Session() as sess:
    placeholder = tf.raw_ops.Placeholder(dtype=tf.float32, shape=[3, 2], name='x')
    variable = tf.raw_ops.VariableV2(shape=[2], dtype=tf.float32, name='variable_yy2')
    value1 = tf.constant([1, 2], dtype=tf.float32)
    assign = tf.raw_ops.Assign(ref=variable, value=value1)
    mul = tf.raw_ops.Mul(x=placeholder, y=variable, name='mul')
    tf.compat.v1.global_variables_initializer()
    tf.io.write_graph(sess.graph, os.path.join(sys.argv[1], "unitialized_variablev2"),
                      "unitialized_variablev2.pb", False)
