# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.moc_frontend.shape_utils import get_static_shape
from openvino.tools.mo.utils.versions_checker import get_environment_setup  # pylint: disable=no-name-in-module
from openvino.tools.mo.utils.error import Error
from distutils.version import LooseVersion
import logging as log


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


def type_supported_by_tf_fe(input_model):
    import tensorflow as tf
    if isinstance(input_model, tf.Graph):
        return True
    elif isinstance(input_model, tf.types.experimental.ConcreteFunction):
        return True
    elif model_is_graph_iterator(input_model):
        return True
    raise False


def create_tf_graph_iterator(input_model):
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
    model = argv['input_model']
    import tensorflow as tf
    from tensorflow.python.training.tracking.base import Trackable
    env_setup = get_environment_setup("tf")
    if isinstance(model, (tf.Graph, tf.compat.v1.GraphDef)):
        return True
    if isinstance(model, tf.compat.v1.Session):
        argv['input_model'] = model.graph
        return True
    if env_setup["tensorflow"] >= LooseVersion("2.6.0") and isinstance(model, (tf.types.experimental.GenericFunction,
                                                                               tf.types.experimental.ConcreteFunction)):
        return True
    if isinstance(model, tf.train.Checkpoint):
        if isinstance(model.root, tf.keras.Model):
            argv['input_model'] = model.root
            return True
        else:
            raise Error("Unknown checkpoint format.")

    if isinstance(model, (tf.keras.layers.Layer, tf.Module, tf.keras.Model)):
        return True
    if isinstance(model, Trackable):
        if hasattr(model, 'signatures') and len(model.signatures.items()):
            if 'serving_default' in model.signatures:
                argv['input_model'] = model.signatures['serving_default']
            elif 'default' in model.signatures:
                argv['input_model'] = model.signatures['default']
            else:
                for signature_name, signature in model.signatures.items():
                    argv['input_model'] = model.signatures[signature_name]
                    log.warning("Could not find the default signature. "
                                "The following signature was used for conversion: {}".format(signature_name))
                    break

        elif hasattr(model, 'graph'):
            argv['input_model'] = model.graph
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
