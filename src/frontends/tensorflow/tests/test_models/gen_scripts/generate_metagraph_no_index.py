# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Build a MetaGraph that contains RestoreV2 -> AssignVariableOp nodes (so a
# non-empty variables map is resolved during loading), but export only the
# ".meta" file without saving a checkpoint. This way no variables index
# (".index") file is produced next to the model, reproducing a MetaGraph
# without a variables index.
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    z_value = [[2., 2., 1.], [1., 1., 2.]]
    x_value = [[1., 2., 3.], [3., 2., 1.]]

    # Main computation depends only on a placeholder and a constant, so it can
    # be converted without any restored variable values.
    tf_y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[2, 3], name='y')
    tf_z = tf.constant(z_value)
    tf.add(tf_y, tf_z, name="AddOperation")

    # A resource variable with RestoreV2 -> AssignVariableOp nodes. These are not
    # connected to the model output but still populate the variables map.
    tf_x = tf.Variable(x_value)
    sess.run(tf.compat.v1.global_variables_initializer())
    var_handle = tf.compat.v1.get_default_graph().get_tensor_by_name("Variable:0")
    prefix = tf.constant("no_such_checkpoint")
    restorev2 = tf.raw_ops.RestoreV2(prefix=prefix, tensor_names=["Variable"],
                                     shape_and_slices=[""], dtypes=[tf.float32],
                                     name="save/RestoreV2/Direct")
    tf.compat.v1.raw_ops.AssignVariableOp(resource=var_handle, value=restorev2[0])

    os.makedirs(os.path.join(sys.argv[1], "metagraph_no_index"))
    # Export only the MetaGraph (*.meta) without saving a checkpoint, so no
    # variables index (*.index) file is produced next to the model.
    tf.compat.v1.train.export_meta_graph(
        filename=os.path.join(sys.argv[1], "metagraph_no_index", "graph.meta"))
