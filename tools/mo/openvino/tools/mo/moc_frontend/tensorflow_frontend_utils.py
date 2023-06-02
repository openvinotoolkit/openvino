# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.moc_frontend.shape_utils import get_static_shape


def trace_tf_model_if_needed(argv):
    import tensorflow as tf
    if not isinstance(argv.input_model, (tf.keras.layers.Layer, tf.Module, tf.keras.Model)):
        return
    argv.input_model = trace_tf_model(argv.input_model, argv.placeholder_shapes, argv['example_input'] if 'example_input' in argv else None)


def trace_tf_model(model, input_shapes, example_input):
    import tensorflow as tf
    if isinstance(model.__call__, tf.types.experimental.GenericFunction):
        tf_function = model.__call__
    else:
        # Wrap model to tf.Function
        @tf.function
        def tf_function(args):
            return model(*args)

    if hasattr(model, '_build_input_shape') and model._build_input_shape is not None:
        if isinstance(model._build_input_shape, list):
            input_spec = [[tf.TensorSpec(shape) for shape in model._build_input_shape]]
        else:
            input_spec = [tf.TensorSpec(model._build_input_shape)]
    elif input_shapes is not None:
        if isinstance(input_shapes, list):
            input_spec = [tf.TensorSpec(get_static_shape(shape)) for shape in input_shapes]
        else:
            input_spec = [tf.TensorSpec(get_static_shape(input_shapes))]
    else:
        input_spec = [tf.TensorSpec(None)]

    try:
        # Trace the model
        concrete_func = tf_function.get_concrete_function(input_spec)
    except:
        if example_input is None:
            raise Exception("Could not trace the TF model. Please provide 'example_input'.")

        try:
            concrete_func = tf_function.get_concrete_function(example_input)
        except Exception as e:
            raise Exception("Could not trace the TF model with the following error: {}".format(e))

    return concrete_func
