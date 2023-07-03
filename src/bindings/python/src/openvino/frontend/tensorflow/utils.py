# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors


from openvino.tools.ovc.moc_frontend.shape_utils import get_static_shape
from openvino.tools.ovc.environment_setup_utils import get_environment_setup  # pylint: disable=no-name-in-module
from openvino.tools.ovc.error import Error
from distutils.version import LooseVersion
import logging as log


def trace_tf_model_if_needed(input_model, placeholder_shapes, placeholder_data_types, example_input):
    import tensorflow as tf
    if not isinstance(input_model, (tf.keras.layers.Layer, tf.Module, tf.keras.Model, tf.types.experimental.GenericFunction)):
        return input_model
    return trace_tf_model(input_model, placeholder_shapes, placeholder_data_types, example_input)


def get_input_spec_from_model(model):
    import tensorflow as tf
    if hasattr(model, "_build_input_shape") and model._build_input_shape is not None:
        if isinstance(model._build_input_shape, list):
            input_spec = [[tf.TensorSpec(shape) for shape in model._build_input_shape]]
        else:
            input_spec = [tf.TensorSpec(model._build_input_shape)]
    else:
        input_spec = [tf.TensorSpec(None)]
    return input_spec


def create_example_input_by_user_shapes(input_shapes, input_types):
    import tensorflow as tf
    if input_shapes is None:
        return None
    if isinstance(input_shapes, dict):
        res = {}
        for name, shape in input_shapes.items():
            shape = get_static_shape(shape, 1)
            args = {}
            if name in input_types:
                args["dtype"] = input_types[name]
            tensor = tf.zeros(shape=shape, **args)
            res[name] = tensor
        return res
    elif isinstance(input_shapes, list):
        res = []
        for idx, shape in enumerate(input_shapes):
            shape = get_static_shape(shape, 1)
            args = {}
            if idx < len(input_types):
                args["dtype"] = input_types[idx]
            tensor = tf.zeros(shape=shape, **args)
            res.append(tensor)
        return res
    raise Error("Could not create example input by provided shape {}".format(input_shapes))


def get_concrete_func(tf_function, example_input, input_needs_packing, error_message, use_example_input=True):
    """
    Runs tracing of TF function and returns a concrete function.

    :param tf_function: TF function that needs to be traced.
    :param example_input: Example of function input.
    :param input_needs_packing: determines if input needs to be packed in a list before passing to TF function.
    It is used when original function was wrapped in outer TF function, which changes function signature.
    In this case wrapper TF function always expects list of inputs which are unpacked inside subfunction.
    So list/tuple are treated as multiple inputs of original model.
    Non list/tuple are treated as single input, and it needs packing to a list,
    as wrapper function always expect list of inputs.
    :param error_message: Error message which should be shown in case of tracing error.
    :param use_example_input: Determines if example_input should be used.

    :returns: Object of type tf.types.experimental.ConcreteFunction.
    """
    if input_needs_packing and not isinstance(example_input, (list, tuple)):
        example_input = [example_input]
    try:
        if use_example_input:
            if not input_needs_packing and isinstance(example_input, (list, tuple)):
                concrete_func = tf_function.get_concrete_function(*example_input)
            else:
                concrete_func = tf_function.get_concrete_function(example_input)

        else:
            concrete_func = tf_function.get_concrete_function()
    except Exception as e:
        raise Exception(error_message.format(e))
    return concrete_func


def trace_tf_model(model, input_shapes, input_types, example_input):
    import tensorflow as tf
    if isinstance(model.__call__, tf.types.experimental.GenericFunction):
        tf_function = model.__call__
        input_needs_packing = False
    elif isinstance(model, tf.types.experimental.GenericFunction):
        tf_function = model
        input_needs_packing = False
    else:
        # Wrap model to tf.Function.
        # In this case we loose input/output tensor names.
        @tf.function
        def tf_function(args):
            return model(*args)
        input_needs_packing = True

    if example_input is not None:
        concrete_func = get_concrete_func(tf_function, example_input, input_needs_packing,
                                          "Could not trace the TF model with the following error: {}")
    elif input_shapes is not None:
        inp = create_example_input_by_user_shapes(input_shapes, input_types)
        concrete_func = get_concrete_func(tf_function, inp, input_needs_packing,
                                          "Could not trace the TF model with the following error: {}")
    else:
        if isinstance(tf_function, tf.types.experimental.GenericFunction) and \
                tf_function.input_signature is not None:
            concrete_func = get_concrete_func(tf_function, None, input_needs_packing,
                                              "Could not trace the TF model with the following error: {}",
                                              use_example_input=False)
        else:
            input_spec = get_input_spec_from_model(model)
            concrete_func = get_concrete_func(tf_function, input_spec, input_needs_packing,
                                              "Could not trace the TF model with the following error: {}.\n"
                                              "Please provide 'example_input'.")

    return concrete_func


def type_supported_by_tf_fe(input_model):
    import tensorflow as tf
    # Types that require tracing
    if isinstance(input_model, (tf.keras.layers.Layer, tf.Module, tf.keras.Model, tf.types.experimental.GenericFunction)):
        return True
    # Types that do not require tracing
    if isinstance(input_model, (tf.Graph, tf.types.experimental.ConcreteFunction)):
        return True
    # GraphIterator
    elif model_is_graph_iterator(input_model):
        return True
    return False


def create_tf_graph_iterator(input_model, placeholder_shapes, placeholder_data_types, example_input):
    input_model = trace_tf_model_if_needed(input_model, placeholder_shapes, placeholder_data_types, example_input)

    import tensorflow as tf
    from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
    if model_is_graph_iterator(input_model):
        return input_model
    if isinstance(input_model, tf.Graph):
        return GraphIteratorTFGraph(input_model)
    elif isinstance(input_model, tf.types.experimental.ConcreteFunction):
        return GraphIteratorTFGraph(input_model.graph)
    raise Exception("Could not wrap model of type {} to GraphIteratorTFGraph.".format(type(input_model)))


def extract_model_graph(argv):
    model = argv["input_model"]
    import tensorflow as tf
    trackable_is_imported = False
    try:
        from tensorflow.python.training.tracking.base import Trackable # pylint: disable=no-name-in-module,import-error
        trackable_is_imported = True
    except:
        log.warning("Could not import tensorflow.python.training.tracking.base.Trackable type.")
    env_setup = get_environment_setup("tf")
    if isinstance(model, tf.Graph):
        return True
    if isinstance(model, tf.compat.v1.GraphDef):
        graph = tf.Graph()
        with graph.as_default():
            tf.graph_util.import_graph_def(model)
        argv["input_model"] = graph
        return True
    if isinstance(model, tf.compat.v1.Session):
        argv["input_model"] = model.graph
        return True
    if env_setup["tensorflow"] >= LooseVersion("2.6.0") and isinstance(model, (tf.types.experimental.GenericFunction,
                                                                               tf.types.experimental.ConcreteFunction)):
        return True
    if isinstance(model, tf.train.Checkpoint):
        if isinstance(model.root, tf.keras.Model):
            argv["input_model"] = model.root
            return True
        else:
            raise Error("Unknown checkpoint format.")

    if isinstance(model, (tf.keras.layers.Layer, tf.Module, tf.keras.Model)):
        return True
    if trackable_is_imported and isinstance(model, Trackable):
        if hasattr(model, "signatures") and len(model.signatures.items()):
            if "serving_default" in model.signatures:
                argv["input_model"] = model.signatures["serving_default"]
            elif "default" in model.signatures:
                argv["input_model"] = model.signatures["default"]
            else:
                for signature_name, signature in model.signatures.items():
                    argv["input_model"] = model.signatures[signature_name]
                    log.warning("Could not find the default signature. "
                                "The following signature was used for conversion: {}".format(signature_name))
                    break

        elif hasattr(model, "graph"):
            argv["input_model"] = model.graph
        else:
            raise Error("Could not find signature of graph in a Trackable object.")
        return True
    if model_is_graph_iterator(model):
        return True
    return False


def model_is_graph_iterator(model):
    try:
        from openvino.frontend.tensorflow.graph_iterator import GraphIteratorTFGraph
    except:
        return False
    return isinstance(model, GraphIteratorTFGraph)
