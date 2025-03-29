# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import tensorflow as tf


# This test model aims to cover conversion of SavedModel with integer input types
class ModelWithGather(tf.Module):
    def __init__(self):
        super(ModelWithGather, self).__init__()
        self.var1 = tf.Variable(5.0)

    @tf.function(input_signature=[tf.TensorSpec([20, 5], tf.float32), tf.TensorSpec([4], tf.int32)])
    def __call__(self, params, indices):
        gather = tf.raw_ops.GatherV2(params=params, indices=indices, axis=0)
        return {'test_output_name': gather * self.var1}


module = ModelWithGather()
tf.saved_model.save(module, os.path.join(sys.argv[1], "saved_model_with_gather"))
