# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    x_value = [[1.,2.,3.],[3.,2.,1.]]
    z_value = [[2.,2.,1.],[1.,1.,2.]]
    tf_x = tf.Variable(x_value)
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1], name='y')
    tf_z = tf.constant(z_value)
    tf_add = tf.add(tf_x, tf_y, name="AddOperation")
    tf_identity = tf.identity(tf_add, name="AddIdentity")
    tf.subtract(tf_identity, tf_z, name="SubOperation")
    sess.run(tf.compat.v1.global_variables_initializer())
    # Produces RestoreV2 -> Identity -> AssignVariableOp
    saver = tf.compat.v1.train.Saver([tf_x])

    input_name = tf.compat.v1.get_default_graph().get_tensor_by_name("save/Const:0")
    var_handle = tf.compat.v1.get_default_graph().get_tensor_by_name("Variable:0")

    # Produces RestoreV2 -> Pack -> AssignVariableOp
    restorev2 = tf.raw_ops.RestoreV2(prefix=input_name, tensor_names=["Variable"], shape_and_slices=[""], dtypes=[tf.float32], name="save/RestoreV2/wPack")
    assign_var = tf.raw_ops.AssignVariableOp(resource = var_handle, value = restorev2)

    # Produces RestoreV2 -> AssignVariableOp
    restorev2 = tf.raw_ops.RestoreV2(prefix=input_name, tensor_names=["Variable"], shape_and_slices=[""], dtypes=[tf.float32], name="save/RestoreV2/Direct")
    assign_var = tf.compat.v1.raw_ops.AssignVariableOp(resource = var_handle, value = restorev2[0])

    os.makedirs(os.path.join(sys.argv[1], "metagraph_variables"))
    saver.save(sess, os.path.join(sys.argv[1], "metagraph_variables", "graph"))
