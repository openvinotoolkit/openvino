# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow as tf

tf.reset_default_graph()
# Create the graph and model
with tf.Session() as sess:
    tensor = tf.placeholder(tf.float32, [2, 3, 5], name='data')
    element_shape = tf.constant([2, 3, 5], dtype=tf.int32)
    max_num_elements = tf.constant(10, dtype=tf.int32)
    empty_tensor_list = tf.raw_ops.EmptyTensorList(element_shape=element_shape, max_num_elements=max_num_elements,
                                                   element_dtype=tf.float32)
    tensor_list_push_back = tf.raw_ops.TensorListPushBack(input_handle=empty_tensor_list, tensor=tensor)
    tensor_list_stack = tf.raw_ops.TensorListStack(input_handle=tensor_list_push_back, element_shape=element_shape,
                                                   element_dtype=tf.float32)
    tf.global_variables_initializer()
    tf.io.write_graph(sess.graph, '.', 'empty_tensor_list.pb', as_text=False)
