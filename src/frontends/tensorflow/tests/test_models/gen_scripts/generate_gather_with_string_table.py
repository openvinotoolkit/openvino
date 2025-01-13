# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

tf.compat.v1.reset_default_graph()
# Create the graph and model
with tf.compat.v1.Session() as sess:
    params = tf.constant(["First sentence", "Second sentence sentence", "Third"], dtype=tf.string)
    indices = tf.compat.v1.placeholder(tf.int32, [2, 3, 5], name='data')
    axes = tf.constant([0], dtype=tf.int32)
    gather = tf.raw_ops.GatherV2(params=params, indices=indices, axis=0)
    tf.compat.v1.global_variables_initializer()
    tf.io.write_graph(sess.graph, os.path.join(sys.argv[1], "gather_with_string_table"),
                      "gather_with_string_table.pb", False)
