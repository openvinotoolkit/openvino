# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

export_dir = os.path.join(sys.argv[1], "saved_model_multi-graph")

#Slash replacing required because otherwise fails on Windows
builder = tf.compat.v1.saved_model.Builder(export_dir if os.name != 'nt' else export_dir.replace("/", "\\"))

# Create the graph and model
with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x_value = [[1.,2.,3.],[3.,2.,1.]]
    z_value = [[2.,2.,1.],[1.,1.,2.]]
    tf_x = tf.compat.v1.Variable(x_value, name="custom_variable_name")
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='y')
    tf_z = tf.constant(z_value)
    tf_add = tf.add(tf_x, tf_y, name="AddOperation")
    tf_identity = tf.identity(tf_add, name="AddIdentity")
    tf.subtract(tf_identity, tf_z, name="SubOperation")
    sess.run(tf.compat.v1.global_variables_initializer())

    builder.add_meta_graph_and_variables(sess, ["train"])

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x_value = [[1.,2.,3.],[3.,2.,1.]]
    tf_x = tf.compat.v1.Variable(x_value, name="custom_variable_name")
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='y')
    tf_add = tf.add(tf_x, tf_y, name="AddOperation")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=None, defer_build=True)
    builder.add_meta_graph([], saver=saver)

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    x_value = [[1.,2.,3.],[3.,2.,1.]]
    tf_x = tf.compat.v1.Variable(x_value, name="custom_variable_name")
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='y')
    tf_add = tf.subtract(tf_x, tf_y, name="SubOperation")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=None, defer_build=True)
    builder.add_meta_graph(["test","test2"], saver=saver)

builder.save()
