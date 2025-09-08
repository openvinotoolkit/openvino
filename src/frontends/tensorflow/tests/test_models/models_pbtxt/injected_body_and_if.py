# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import function

tf.reset_default_graph()


@function.Defun(tf.int32)
def aux_func(x):
    return x ** 2


@function.Defun(tf.int32, tf.int32)
def then_branch_func(x, y):
    return x + y


@function.Defun(tf.int32, tf.int32)
def else_branch_func(x, y):
    return aux_func(x - y)


with tf.Session() as sess:
    x = tf.placeholder(tf.int32, [2], 'x')
    y = tf.placeholder(tf.int32, [1], 'y')
    const_cond = tf.constant(10, dtype=tf.int32)
    cond = tf.raw_ops.Greater(x=x, y=const_cond)
    if_op = tf.raw_ops.If(cond=cond, input=[x, y], Tout=[tf.int32], then_branch=then_branch_func,
                          else_branch=else_branch_func)
    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'injected_body_and_if.pbtxt', as_text=True)
