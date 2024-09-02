# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
class SampleGraph(tf.Module):
  def __init__(self):
    super(SampleGraph, self).__init__()
    self.var1 = tf.constant([[[[0,0,1,0],[0,0,0,0],[0,2,1,0]],[[0,0,0,0],[0,1,0,0],[2,0,0,0]]]], dtype=tf.float32)
  @tf.function(input_signature=[tf.TensorSpec([1,2,3,3], tf.float32)])
  def __call__(self, x):
    conv = tf.raw_ops.Conv2D(input=x, filter=self.var1, strides=[1,1,1,1], padding="VALID")
    return {'test_output_name': conv}

module = SampleGraph()
sm_path = os.path.join(sys.argv[1], "densify")
tf.saved_model.save(module, sm_path)

converter = tf.lite.TFLiteConverter.from_saved_model(sm_path) # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT, tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(sys.argv[1], sm_path + ".tflite"), 'wb') as f:
  f.write(tflite_model)