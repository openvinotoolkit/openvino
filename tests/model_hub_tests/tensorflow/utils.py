# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

type_map = {
    tf.float64: np.float64,
    tf.float32: np.float32,
    tf.int8: np.int8,
    tf.int16: np.int16,
    tf.int32: np.int32,
    tf.int64: np.int64,
    tf.uint8: np.uint8,
    tf.uint16: np.uint16,
    tf.string: str,
    tf.bool: bool,
}


def get_input_info(input_tensor, input_name):
    input_shape = []
    try:
        for dim in input_tensor.shape.as_list():
            if dim is None:
                input_shape.append(1)
            else:
                input_shape.append(dim)
    except ValueError:
        # unknown rank case
        pass
    assert input_tensor.dtype in type_map, "Unsupported input type: {}".format(input_tensor.dtype)
    return input_name, input_shape, type_map[input_tensor.dtype]


def load_graph(graph_filename):
    with tf_v1.gfile.GFile(graph_filename, "rb") as f:
        graph_def = tf_v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf_v1.Graph().as_default() as graph:
        tf_v1.import_graph_def(graph_def, name="")
    return graph


def get_input_signature(graph: tf_v1.Graph):
    input_signature = []
    for op in graph.get_operations():
        if op.type == "Placeholder":
            op_name = op.name + ':0'
            op_shape = tf.TensorShape(op.node_def.attr['shape'].shape)
            op_type = tf.dtypes.DType(op.node_def.attr['dtype'].type)
            input_signature.append((op_name, tf.TensorSpec(op_shape, op_type)))
    return input_signature


def children(op, graph):
    op = graph.get_operation_by_name(op)
    return set(op for out in op.outputs for op in out.consumers())


def collect_control_dependencies(graph):
    control_dependents_map = {}
    for op in graph.get_operations():
        for control_input in op.control_inputs:
            if control_input.name not in control_dependents_map:
                control_dependents_map[control_input.name] = [op]
            else:
                control_dependents_map[control_input.name].append(op)
    return control_dependents_map


def get_output_signature(graph: tf_v1.Graph):
    outputs = []
    unlikely_output_types = ['Const', 'Assign', 'NoOp', 'Placeholder', 'Assert',
                             'switch_t', 'switch_f', 'TensorArrayCloseV3']
    control_dependents_map = collect_control_dependencies(graph)
    for op in graph.get_operations():
        if len(children(op.name, graph)) == 0 and op.name not in control_dependents_map:
            if op.type not in unlikely_output_types:
                outputs.append(op.name + ':0')
    return outputs
