# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf


@tf.function
def second_func(x):
    x = tf.raw_ops.TopKV2(input=x, k=tf.constant(3, tf.int32))[1]
    x = tf.add(x, tf.constant(10, tf.int32))
    return x


@tf.function
def first_func(x, y):
    return second_func(x - y)


graph_def = first_func.get_concrete_function(tf.constant([1, 2, 3, 4, 5], dtype=tf.int32),
                                             tf.constant([0, 1, 1, 1, 1], dtype=tf.int32)).graph.as_graph_def()
tf.io.write_graph(graph_def, '.', 'partitioned_call2.pbtxt', as_text=True)
