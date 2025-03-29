# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf


# This test model aims to cover conversion of SavedModel with multiple tensor names for output
# among which TF FE needs to remain only user specific names and exclude internal tensor names
class ModelParameterResult(tf.Module):
    def __init__(self):
        super(ModelParameterResult, self).__init__()

    @tf.function(input_signature=[tf.TensorSpec([20, 5], tf.float32)])
    def __call__(self, params):
        identity1 = tf.raw_ops.Identity(input=params)
        identity2 = tf.raw_ops.Identity(input=identity1)
        identity3 = tf.raw_ops.Identity(input=identity2)
        return {'test_output_name': identity3}


module = ModelParameterResult()
tf.saved_model.save(module, os.path.join(sys.argv[1], "saved_model_parameter_result"))
