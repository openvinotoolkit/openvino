# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Generates two TFLite test models exercising the SLICE op:
#   - slice_const_size.tflite : size operand is a fully non-negative Constant.
#                                The translator must skip the ShapeOf+Select
#                                cascade emitted for size=-1, so the Slice
#                                output shape is statically inferable.
#   - slice_neg_size.tflite   : size operand contains -1 ("to end").
#                                The translator must keep the ShapeOf+Select
#                                cascade alive (Slice's `stop` input is fed
#                                by a Select node).

import os
import sys

# Silence verbose TensorFlow logs to keep build output clean.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


def build_and_save(out_dir, name, begin, size, input_shape=(1, 128, 8, 256)):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        x = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
        tf.slice(x, begin, size, name='sliced')
        tf_net = sess.graph_def

    pb_name = name + '.pb'
    tflite_name = name + '.tflite'
    tf.io.write_graph(tf_net, out_dir, pb_name, False)

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        os.path.join(out_dir, pb_name), ['input'], ['sliced'])
    tflite_model = converter.convert()

    with tf.io.gfile.GFile(os.path.join(out_dir, tflite_name), 'wb') as f:
        f.write(tflite_model)


out_dir = sys.argv[1]
build_and_save(out_dir, 'slice_const_size', begin=[0, 0, 0, 0], size=[1, 128, 4, 128])
build_and_save(out_dir, 'slice_neg_size',   begin=[0, 0, 0, 0], size=[1, 128, 4, -1])
