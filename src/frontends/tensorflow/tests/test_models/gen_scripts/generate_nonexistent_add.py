# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# unsupported add tensorflow model generator
#

import numpy as np
import os
import sys
import tensorflow as tf


def main():
    tf.compat.v1.reset_default_graph()

    # Create the graph and model
    with tf.compat.v1.Session() as sess:
        const2 = tf.constant(2.0, dtype=tf.float32)
        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 3], name='x')
        relu = tf.nn.relu(x)    
        add = tf.add(relu, const2, name="add")
        tf.multiply(add, relu)

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "nonexistent_add"), "nonexistent_add.pb", False)

    with open(os.path.join(sys.argv[1], "nonexistent_add", "nonexistent_add.pb"), mode='rb') as file:
        modelContent = file.read()

    modelContent = modelContent.replace(b"AddV2", b"Adddd")

    with open(os.path.join(sys.argv[1], "nonexistent_add", "nonexistent_add.pb"), mode='wb') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()
