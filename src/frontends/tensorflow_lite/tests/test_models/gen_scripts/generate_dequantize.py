# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
class SampleGraph(tf.Module):
  def __init__(self):
    super(SampleGraph, self).__init__()
    self.var1 = tf.constant([[1, 0.75],[2000.43, -0.12345]], dtype=tf.float32)
  @tf.function(input_signature=[tf.TensorSpec([2,2], tf.float32)])
  def __call__(self, x):
    res = self.var1 + x
    return {'test_output_name': res}

module = SampleGraph()
sm_path = os.path.join(sys.argv[1], "dequantize")
tf.saved_model.save(module, sm_path)

converter = tf.lite.TFLiteConverter.from_saved_model(sm_path) # path to the SavedModel directory
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the model.
with open(os.path.join(sys.argv[1], sm_path + ".tflite"), 'wb') as f:
  f.write(tflite_model)