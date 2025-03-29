# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf

# Create the graph and model
class AddVariable(tf.Module):
  def __init__(self):
    super(AddVariable, self).__init__()
    self.var1 = tf.Variable(123.0)
  @tf.function(input_signature=[tf.TensorSpec([1], tf.float32)])
  def __call__(self, x):
    return {'test_output_name': x * self.var1}

module = AddVariable()
tf.saved_model.save(module, os.path.join(sys.argv[1], "saved_model_variables"))
