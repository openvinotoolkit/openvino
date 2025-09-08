# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# unsupported relu tensorflow model generator
#

import numpy as np
import os
import sys
import tensorflow as tf


def main():
    tf.compat.v1.reset_default_graph()

    # Create the graph and model
    with tf.compat.v1.Session() as sess:
        input = tf.compat.v1.placeholder(tf.float32, [1, 3, 3, 1], 'x')

        tf.nn.relu(input, name="relu_0")

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "relu_unsupported"), "relu_unsupported.pb", False)

    with open(os.path.join(sys.argv[1], "relu_unsupported", "relu_unsupported.pb"), mode='rb') as file:
        modelContent = file.read()

    modelContent = modelContent.replace(b"Relu", b"Rxyz")

    with open(os.path.join(sys.argv[1], "relu_unsupported", "relu_unsupported.pb"), mode='wb') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()
