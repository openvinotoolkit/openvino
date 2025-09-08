# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# model with Assert node generator
#

import os
import sys

import numpy as np
import tensorflow as tf


def main():
    tf.compat.v1.reset_default_graph()

    # Create the graph and model
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='x')
        y = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='y')
        tf.raw_ops.AddV2(x=x, y=y)
        shape1 = tf.raw_ops.Shape(input=x)
        shape2 = tf.raw_ops.Shape(input=y)
        equal = tf.raw_ops.Equal(x=shape1, y=shape2)
        axis = tf.constant([0], dtype=tf.int32)
        all_equal = tf.raw_ops.All(input=equal, axis=axis)
        message = tf.constant("Shapes of operands are incompatible", dtype=tf.string)
        tf.raw_ops.Assert(condition=all_equal, data=[message])

        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "model_with_assert"), "model_with_assert.pb", False)


if __name__ == "__main__":
    main()
