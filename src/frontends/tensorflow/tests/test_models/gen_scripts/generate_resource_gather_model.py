# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import tensorflow as tf


def main():
    # this test model imitates a sub-graph of one model with ResourceGather from our model scope
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        iteratorv2 = tf.raw_ops.IteratorV2(shared_name="iterator", container="container",
                                           output_types=[tf.float32, tf.float32], output_shapes=[[2, 3], [3]])
        iterator_get_next = tf.raw_ops.IteratorGetNext(iterator=iteratorv2, output_types=[tf.float32, tf.float32],
                                                       output_shapes=[[2, 3], [3]])
        cast1 = tf.raw_ops.Cast(x=iterator_get_next[0], DstT=tf.int32)
        cast2 = tf.raw_ops.Cast(x=iterator_get_next[1], DstT=tf.int32)

        table1 = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
        table2 = tf.constant([10, 11, 12, 13, 14], dtype=tf.float32)

        resource_gather1 = tf.raw_ops.Gather(params=table1, indices=cast1,
                                             name="embedding_lookup1")
        resource_gather2 = tf.raw_ops.Gather(params=table2, indices=cast2,
                                             name="embedding_lookup2")

        tf.raw_ops.Mul(x=resource_gather1, y=resource_gather2)
        tf.compat.v1.global_variables_initializer()
        tf_net = sess.graph_def

    tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "resource_gather_model"), "resource_gather_model.pbtxt", True)

    with open(os.path.join(sys.argv[1], "resource_gather_model", "resource_gather_model.pbtxt"),
              mode='r') as file:
        modelContent = file.read()

    # using latest TensorFlow version, it is not possible to generate a model with ResourceGather
    # due to explicit use of resource, so we have to replace Gather with ResourceGather
    # in earlier versions of TensorFlow, the semantics was the same for these operations
    # we have an example of the model in our scope
    modelContent = modelContent.replace("Gather", "ResourceGather")

    with open(os.path.join(sys.argv[1], "resource_gather_model", "resource_gather_model.pbtxt"),
              mode='w') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()
