# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
import sys

# do not print messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.reset_default_graph()

# Create the graph and model
with tf.compat.v1.Session() as sess:
    input1 = tf.compat.v1.placeholder(tf.float32, [1, 3, 3, 1], 'inputX1')
    kernel1 = tf.constant(np.random.randn(1, 1, 1, 1), dtype=tf.float32)
    conv2d1 = tf.nn.conv2d(input1, kernel1, strides=[1, 1], padding='VALID')
    relu1 = tf.nn.relu6(conv2d1)

    input2 = tf.compat.v1.placeholder(tf.float32, [1, 3, 3, 1], 'inputX2')
    kernel2 = tf.constant(np.random.randn(1, 1, 1, 1), dtype=tf.float32)
    depthconv2d2 = tf.nn.depthwise_conv2d(input2, kernel2, strides=[1, 1, 1, 1], padding='VALID')
    sigmoid2 = tf.nn.sigmoid(depthconv2d2)

    concat = tf.concat([relu1, sigmoid2], axis=-1)

    random_constant = tf.constant(np.random.randn(1, 1, 1, 1), dtype=tf.float32)
    add1 = tf.add(concat, random_constant)
    sig = tf.nn.relu(add1, name="sigmoid3b")
    paddings = tf.constant([[0, 0], [1, 1], [2, 2], [0, 0]])
    pad = tf.pad(add1, paddings, "CONSTANT", name="pad")


    tf.compat.v1.global_variables_initializer()
    tf_net = sess.graph_def

path_to_model_dir = os.path.join(sys.argv[1], "2in_2out")
tf_file_name = '2in_2out.pb'
tflite_file_name = '2in_2out.tflite'
tf.io.write_graph(tf_net, path_to_model_dir, tf_file_name, False)

inputs = ["inputX1", "inputX2"]
outputs = ["pad", "sigmoid3b"]

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(os.path.join(path_to_model_dir, tf_file_name), inputs, outputs)
tflite_model = converter.convert()

tflite_model_path = os.path.join(path_to_model_dir, tflite_file_name)
with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

