# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf
import numpy as np

# Create the graph and model
tf.compat.v1.reset_default_graph()
with tf.compat.v1.Session() as sess:
    tf_x = tf.constant(np.array([[1, 2]], dtype=np.dtype('i8')))
    tf_cast = tf.raw_ops.Cast(x=-1, DstT=tf.int64)
    tf_fill = tf.fill([0, 2], tf_cast, name="FillOperation")
    tf_fill2 = tf.fill([1, 2], tf_cast, name="FillOperation2")
    tf_output = tf.raw_ops.ConcatV2(values=[tf_x, tf_fill, tf_fill2], axis=0, name="ConcatOperation")
    tf.compat.v1.saved_model.simple_save(sess, os.path.join(sys.argv[1], "saved_model_broadcast_issue"), inputs={'x':tf_x}, outputs={'output':tf_output})
