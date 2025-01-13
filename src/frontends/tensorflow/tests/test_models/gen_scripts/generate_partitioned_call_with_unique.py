# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow.compat.v1 as tf


def main():
    tf.compat.v1.reset_default_graph()

    @tf.function
    def second_func(x):
        relu = tf.raw_ops.Relu(features=x)
        y, idx = tf.raw_ops.Unique(x=relu)
        return y, idx

    @tf.function
    def first_func(x):
        y, idx = second_func(x)
        sigmoid = tf.raw_ops.Sigmoid(x=y)
        const_one = tf.constant(1, dtype=tf.int32)
        add = tf.raw_ops.AddV2(x=idx, y=const_one)
        return sigmoid, add

    tf_net = first_func.get_concrete_function(tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)).graph.as_graph_def()
    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "partitioned_call_with_unique"),
                      "partitioned_call_with_unique.pb", False)


if __name__ == "__main__":
    main()
