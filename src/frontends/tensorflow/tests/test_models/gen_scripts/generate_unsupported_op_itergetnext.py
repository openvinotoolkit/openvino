# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import sys
import tensorflow as tf


def main():
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        iteratorv2 = tf.raw_ops.IteratorV2(shared_name="iterator", container="container",
                                           output_types=[tf.float32, tf.float32], output_shapes=[[2, 3], [3]])
        iterator_get_next = tf.raw_ops.IteratorGetNext(iterator=iteratorv2, output_types=[tf.float32, tf.float32],
                                                       output_shapes=[[2, 3], [3]])

        tf.raw_ops.AddV2(x=iterator_get_next[0], y=iterator_get_next[1])
        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "unsupported_op_itergetnext"), "unsupported_op_itergetnext.pb",
                      False)

    with open(os.path.join(sys.argv[1], "unsupported_op_itergetnext", "unsupported_op_itergetnext.pb"),
              mode='rb') as file:
        modelContent = file.read()

    modelContent = modelContent.replace(b"IteratorV2", b"IterVVVVVV")

    with open(os.path.join(sys.argv[1], "unsupported_op_itergetnext", "unsupported_op_itergetnext.pb"),
              mode='wb') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()
