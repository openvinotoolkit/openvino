# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import tensorflow as tf
from openvino import Core, convert_model


def make_4d_resize_bilinear_tflite(save_path):
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        placeholder = tf.compat.v1.placeholder(tf.float32, [1, 3, 4, 3], name="Input")
        tf.compat.v1.image.resize_bilinear(placeholder, size=[8, 8], name="Output")
        graph_def = sess.graph_def

    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(graph_def, name="")
        input_tensors = [g.get_tensor_by_name("Input:0")]
        output_tensors = [g.get_tensor_by_name("Output:0")]

    tflite_model = tf.compat.v1.lite.TFLiteConverter(
        graph_def, input_tensors=input_tensors, output_tensors=output_tensors
    ).convert()

    tflite_path = os.path.join(save_path, "resize_bilinear_4d.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    return tflite_path


def test_resize_bilinear_4d_uses_supported_interpolate_axes(tmp_path, ie_device, precision):
    tflite_path = make_4d_resize_bilinear_tflite(str(tmp_path))
    model = convert_model(tflite_path)

    interpolate_ops = [op for op in model.get_ordered_ops() if op.get_type_name() == "Interpolate"]
    assert len(interpolate_ops) == 1, f"expected exactly 1 Interpolate op, got {len(interpolate_ops)}"
    interpolate = interpolate_ops[0]

    axes_input = interpolate.input_value(2).get_node()
    assert axes_input.get_type_name() == "Constant", "Interpolate axes input should be a Constant"
    axes_values = list(axes_input.get_vector())
    assert axes_values == [2, 3], (
        f"Interpolate axes should be the natively-supported {{2, 3}} (reached via an "
        f"explicit Transpose to NCHW), got {axes_values}"
    )

    transpose_ops = [op for op in model.get_ordered_ops() if op.get_type_name() == "Transpose"]
    assert len(transpose_ops) == 2, f"expected 2 Transpose ops (NHWC<->NCHW around Interpolate), got {len(transpose_ops)}"


def test_resize_bilinear_4d_constant_folds_correctly(ie_device, precision):
    # Converts the raw TF graph directly, skipping TFLiteConverter (which would pre-fold
    # a constant resize on its own) so that OpenVINO's own ConstantFolding evaluates the
    # Interpolate node during conversion -- the exact path that used to throw
    # "Axes are not correct!" before the fix.
    np.random.seed(0)
    image_data = np.random.rand(1, 3, 4, 3).astype(np.float32)
    placeholder_data = np.array([0.0], dtype=np.float32)

    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session() as sess:
        placeholder = tf.compat.v1.placeholder(tf.float32, [1], name="Input")
        resized = tf.compat.v1.image.resize_bilinear(tf.constant(image_data), size=[8, 8])
        output = tf.add(resized, placeholder, name="Output")
        expected = sess.run(output, feed_dict={placeholder: placeholder_data})
        graph_def = sess.graph_def

    model = convert_model(graph_def)
    compiled = Core().compile_model(model, "CPU")
    result = compiled([placeholder_data])[compiled.output(0)]

    np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)
