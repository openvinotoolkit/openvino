# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import tensorflow as tf

tf.compat.v1.reset_default_graph()

# Create the graph and model
with tf.compat.v1.Session() as sess:
    input1 = tf.compat.v1.placeholder(tf.float32, [2, 3], 'input1')
    input2 = tf.compat.v1.placeholder(tf.float32, [2, 3], 'input2')

    add = tf.add(input1, input2, name="add")
    with tf.control_dependencies([add]):
        sub = tf.subtract(input1, input2, name="sub")

    tf.compat.v1.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "control_dependency"), 'control_dependency.pb', False)
