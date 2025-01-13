# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

tf.compat.v1.reset_default_graph()

# Create the graph and model
with tf.compat.v1.Session() as sess:
    tensor = tf.compat.v1.placeholder(tf.float32, [2, 3, 5], name='data')
    element_shape = tf.constant([2, 3, 5], dtype=tf.int32)
    max_num_elements = tf.constant(10, dtype=tf.int32)
    empty_tensor_list = tf.raw_ops.EmptyTensorList(element_shape=element_shape, max_num_elements=max_num_elements,
                                                   element_dtype=tf.float32)
    tensor_list_push_back = tf.raw_ops.TensorListPushBack(input_handle=empty_tensor_list, tensor=tensor)
    tensor_list_stack = tf.raw_ops.TensorListStack(input_handle=tensor_list_push_back, element_shape=element_shape,
                                                   element_dtype=tf.float32)

    tf.compat.v1.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, os.path.join(sys.argv[1], "empty_tensor_list"), 'empty_tensor_list.pb', False)
