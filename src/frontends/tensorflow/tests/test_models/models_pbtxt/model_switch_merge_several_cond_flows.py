# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Documents how model_switch_merge_several_cond_flows.pbtxt was produced.
# Two Switch nodes: `Switch` feeds real data (AddV2/Sub -> Merge); `Switch_1` is consumed only
# through control dependencies, so the frontend prunes and frees it before Switch/Merge resolution
# and its weak_ptr in the Merge conditional-flow marker legitimately expires. Regenerate by running
# this in a TensorFlow environment.

import numpy as np
import tensorflow.compat.v1 as tf

tf.reset_default_graph()

with tf.Session() as sess:
    x = tf.placeholder(np.float32, [2], 'x')
    cond = tf.constant(True, dtype=tf.bool)
    switch_false, switch_true = tf.raw_ops.Switch(data=x, pred=cond)

    cond2 = tf.constant(True, dtype=tf.bool)
    switch2_false, switch2_true = tf.raw_ops.Switch(data=cond2, pred=cond2)
    with tf.control_dependencies([switch2_true]):
        const_sub = tf.constant(5, dtype=np.float32)
    with tf.control_dependencies([switch2_false]):
        const_add = tf.constant(2, dtype=np.float32)

    add = tf.raw_ops.AddV2(x=switch_false, y=const_add)
    sub = tf.raw_ops.Sub(x=switch_true, y=const_sub)
    merge = tf.raw_ops.Merge(inputs=[add, sub])
    const_main = tf.constant(1, dtype=np.float32)
    tf.raw_ops.AddV2(x=merge[0], y=const_main)
    tf.global_variables_initializer()
    tf_net = sess.graph_def

tf.io.write_graph(tf_net, './', 'model_switch_merge_several_cond_flows.pbtxt', True)
