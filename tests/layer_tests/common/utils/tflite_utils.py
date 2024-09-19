import itertools
import os
import warnings

import numpy as np
import tensorflow as tf

from common.utils.tf_utils import summarize_graph


def make_positive_array(inputs_dict):
    for input in inputs_dict.keys():
        inputs_dict[input] = np.random.randint(1, 10, inputs_dict[input]).astype(np.float32)
    return inputs_dict


def short_range(inputs_dict):
    for input in inputs_dict.keys():
        inputs_dict[input] = np.random.randint(-1, 1, inputs_dict[input]).astype(np.float32)
    return inputs_dict


def make_boolean_array(inputs_dict):
    for input in inputs_dict.keys():
        inputs_dict[input] = np.random.randint(0, 1, inputs_dict[input]) > 1
    return inputs_dict


def make_int32_positive_array(inputs_dict):
    for input in inputs_dict.keys():
        inputs_dict[input] = np.random.randint(1, 10, inputs_dict[input]).astype(np.int32)
    return inputs_dict


data_generators = {
    'positive': make_positive_array,
    'short_range': short_range,
    'boolean': make_boolean_array,
    'int32_positive': make_int32_positive_array,
}


def activation_helper(input_node, activation_name, name):
    if activation_name is None:
        return input_node
    else:
        return activation_name(input_node, name=name)


additional_test_params = [
    [
        {'axis': None},
        {'axis': -1}
    ],
    [
        {'activation': None},
        {'activation': 'tf.nn.relu'},
        {'activation': 'tf.nn.relu6'},
        # skip tanh and signbit since tflite doesn't fuse such activations
        # https://github.com/tensorflow/tensorflow/blob/77d8c333405a080c57850c45531dbbf077b2bd0e/tensorflow/compiler/mlir/lite/transforms/optimize_patterns.td#L86:L89
        # {'activation': tf.math.tanh},
        # {'activation': lambda x, name: tf.identity(tf.experimental.numpy.signbit(x), name=name)},
        {'activation': 'lambda x, name: tf.math.minimum(tf.math.maximum(-1., x), 1., name=name)'}
    ]
]


def save_pb_to_tflite(pb_model):
    graph_summary = summarize_graph(pb_model)
    inputs = [k for k in graph_summary['inputs'].keys()]
    outputs = graph_summary['outputs']

    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(pb_model, inputs, outputs)
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(os.path.dirname(pb_model), 'model.tflite')
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model_path


def get_tflite_results(use_legacy_frontend, inputs_dict, model_path):
    interpreter = tf.compat.v1.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_name_to_id_mapping = {input['name']: input['index'] for input in input_details}

    for layer, data in inputs_dict.items():
        tensor_index = input_name_to_id_mapping[layer]
        tensor_id = next(i for i, tensor in enumerate(input_details) if tensor['index'] == tensor_index)

        if list(input_details[tensor_id]['shape']) != list(data.shape):
            warnings.warn(f'Model and data have different shapes:\nModel {tensor_id} '
                          f'input shape{input_details[tensor_id]["shape"]}\nInput data shape: {data.shape}')
            interpreter.resize_tensor_input(input_details[tensor_id]['index'], data.shape)
            interpreter.allocate_tensors()

        interpreter.set_tensor(input_details[tensor_id]['index'], data)

    interpreter.invoke()
    tf_lite_result = dict()
    for output in output_details:
        tf_lite_result[output['name']] = interpreter.get_tensor(output['index'])

    result = dict()
    for out in tf_lite_result.keys():
        _tf_res = tf_lite_result[out]
        result[out] = _tf_res

    return tf_lite_result


def save_tf2_saved_model_to_tflite(savedmodel):
    import tensorflow as tf

    converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(savedmodel)
    tflite_model = converter.convert()

    tflite_model_path = os.path.join(os.path.dirname(savedmodel), 'model.tflite')
    with tf.io.gfile.GFile(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    return tflite_model_path


def get_tensors_from_graph(graph, ops: list):
    tensors = []
    for input_op in ops:
        input_op_tensors = graph.get_operation_by_name(input_op).outputs
        for op_out_tensor in input_op_tensors:
            tensors.append(op_out_tensor)

    return tensors


def parametrize_tests(lhs, rhs):
    test_data = list(itertools.product(lhs, rhs))
    for i, (parameters, shapes) in enumerate(test_data):
        parameters.update(shapes)
        test_data[i] = parameters.copy()
    return test_data
