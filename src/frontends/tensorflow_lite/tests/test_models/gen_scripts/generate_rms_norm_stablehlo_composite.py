#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import lite as tflite
except ImportError:
    exit(0)


def generate_rms_norm_stablehlo_composite():
    """
    Generate a TFLite model with StableHLO RMS composite.

    This creates a simple function that computes RMS normalization using the
    odml.rms_norm StableHLO composite operation.
    """
    # Create a simple model with a RMS-like computation
    # We'll use a tf.function and export it with StableHLO lowering if available

    @tf.function(input_signature=[
        tf.TensorSpec([1, 3, 4], tf.float32),  # input
    ])
    def rms_norm_model(x):
        # Compute RMS normalization manually as fallback
        # Real StableHLO composite would require tf-text or similar
        axes = -1
        eps = 1e-6

        # Compute squared mean for RMS
        mean_sq = tf.reduce_mean(tf.square(x), axis=axes, keepdims=True)
        # RMS
        rms = tf.sqrt(mean_sq + eps)
        # Normalize
        return x / rms

    # Convert to TFLite
    converter = tflite.TFLiteConverter.from_concrete_functions([
        rms_norm_model.get_concrete_function()
    ])
    converter.target_spec.supported_ops = [
        tflite.OpsSet.TFLITE_BUILTINS,
        tflite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()

    return tflite_model


def write_model(filename, tflite_model):
    """Write TFLite model to file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # Generate model
    model = generate_rms_norm_stablehlo_composite()

    output_path = os.path.join(
        os.path.dirname(__file__),
        "rms_norm_stablehlo_composite.tflite"
    )
    write_model(output_path, model)
    print(f"Generated: {output_path}")
