# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

tf.compat.v1.reset_default_graph()
# Create the graph and model
with tf.compat.v1.Session() as sess:
    sentences = tf.compat.v1.placeholder(tf.string, [2], name='sentences')
    tf.raw_ops.StringLower(input=sentences, name='string_lower')
    tf.compat.v1.global_variables_initializer()
    tf.io.write_graph(sess.graph, os.path.join(sys.argv[1], "string_lower"),
                      "string_lower.pb", False)
