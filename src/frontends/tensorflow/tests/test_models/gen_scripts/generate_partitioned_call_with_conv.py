# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import tensorflow.compat.v1 as tf


def main():
    '''
    This test model aims to test that the conversion for the body graphs is performed with set input shapes
    that allows to get more optimized ov::Model for the body graphs.
    In particular, we check that the resulted graph contains Convolution operation instead of GroupConvolution
    '''
    tf.compat.v1.reset_default_graph()

    @tf.function
    def second_func(input, filter):
        conv = tf.raw_ops.Conv2D(input=input, filter=filter, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
        return conv

    @tf.function
    def first_func(input, filter):
        conv = second_func(input, filter)
        return conv

    input_data = np.random.rand(1, 1, 10, 10)
    filter_data = np.random.rand(3, 3, 1, 1)
    tf_net = first_func.get_concrete_function(tf.constant(input_data, dtype=tf.float32),
                                              tf.constant(filter_data, dtype=tf.float32)).graph.as_graph_def()
    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "partitioned_call_with_conv"),
                      "partitioned_call_with_conv.pb", False)


if __name__ == "__main__":
    main()
