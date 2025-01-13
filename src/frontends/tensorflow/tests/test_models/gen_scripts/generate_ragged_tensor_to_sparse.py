# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow.compat.v1 as tf


def main():
    tf.compat.v1.reset_default_graph()

    @tf.function
    def second_func(strings, row_splits):
        sparse_indices, sparse_values, sparse_dense_shape = tf.raw_ops.RaggedTensorToSparse(
            rt_nested_splits=[row_splits],
            rt_dense_values=strings)
        return sparse_indices, sparse_values, sparse_dense_shape

    @tf.function
    def first_func(strings, row_splits):
        sparse_indices, _, sparse_dense_shape = second_func(strings, row_splits)
        sparse_indices = tf.raw_ops.Reshape(tensor=sparse_indices, shape=[-1])
        sparse_dense_shape = tf.raw_ops.Reshape(tensor=sparse_dense_shape, shape=[-1])
        concat = tf.raw_ops.Concat(concat_dim=0, values=[sparse_indices, sparse_dense_shape])
        return concat

    tf_net = first_func.get_concrete_function(tf.constant(["abc", "bcd", "cc"], dtype=tf.string),
                                              tf.constant([0, 2, 2, 3, 3], dtype=tf.int32)).graph.as_graph_def()
    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "ragged_tensor_to_sparse"),
                      "ragged_tensor_to_sparse.pb", False)


if __name__ == "__main__":
    main()
