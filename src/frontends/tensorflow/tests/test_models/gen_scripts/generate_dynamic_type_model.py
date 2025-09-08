# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# dynamic type is below FW node for unsupported operation
#

import os
import sys

import tensorflow as tf


def main():
    tf.compat.v1.reset_default_graph()

    # Create the graph and model
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 3], name='x')
        relu = tf.raw_ops.Relu(features=x)
        tf.raw_ops.Log1p(x=relu)

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "dynamic_type_model"), "dynamic_type_model.pb", False)

    with open(os.path.join(sys.argv[1], "dynamic_type_model", "dynamic_type_model.pb"), mode='rb') as file:
        modelContent = file.read()

    modelContent = modelContent.replace(b"Relu", b"Rrrr")

    with open(os.path.join(sys.argv[1], "dynamic_type_model", "dynamic_type_model.pb"), mode='wb') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()
