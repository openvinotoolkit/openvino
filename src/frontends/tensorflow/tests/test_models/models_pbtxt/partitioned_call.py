# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf


@tf.function
def second_func(x):
    return x ** 2


@tf.function
def first_func(x, y):
    return second_func(x - y)


graph_def = first_func.get_concrete_function(tf.constant([6, 3]), tf.constant([7])).graph.as_graph_def()
tf.io.write_graph(graph_def, '.', 'partitioned_call.pbtxt', as_text=True)
