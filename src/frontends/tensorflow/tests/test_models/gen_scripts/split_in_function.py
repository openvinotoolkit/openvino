# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow.compat.v1 as tf
import numpy as np


def main():
    tf.compat.v1.reset_default_graph()

    @tf.function
    def second_func(x):
        x1, x2, x3 = tf.split(x, 3)
        return x1 + x2 + x3

    @tf.function
    def first_func(x):
        return second_func(x)

    tf_net = first_func.get_concrete_function(tf.constant(
        np.random.randn(3, 20), dtype=tf.float32)).graph.as_graph_def()
    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "split_in_function"),
                      "split_in_function.pb", False)


if __name__ == "__main__":
    main()
