# Copyright (C) 2018-2025 Intel Corporation
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


def unpack_tf_result(tensor):
    if isinstance(tensor, tf.Tensor):
        return tensor.numpy()
    elif isinstance(tensor, (list, tuple)):
        res = []
        for elem in tensor:
            res.append(unpack_tf_result(elem))
        if isinstance(tensor, list):
            return res
        else:
            return tuple(res)
    elif isinstance(tensor, dict):
        res = {}
        for out_name, out_value in tensor.items():
            res[out_name] = unpack_tf_result(out_value)
        return res
    raise Exception("Unknown output type of original FW inference result: {}".format(type(tensor)))


def repack_ov_result_to_tf_format(ov_out, signature, outer_name=None):
    if signature is None:
        return ov_out

    from tensorflow.python.framework.tensor import TensorSpec
    if isinstance(signature, (tf.Tensor, TensorSpec)):
        out_name = signature.name
        assert out_name in ov_out or outer_name in ov_out, "Could not match OV output and FW signature."
        # Case when ov result has inner tensor name
        if out_name in ov_out:
            return ov_out[out_name]
        # Case when ov result has correct outer name
        if outer_name is not None and outer_name in ov_out:
            return ov_out[outer_name]
        raise Exception("Could not match OV output and FW signature.")
    elif isinstance(signature, (list, tuple)):
        res = []
        for idx, elem in enumerate(signature):
            res.append(repack_ov_result_to_tf_format(ov_out, signature[idx]))
        if isinstance(signature, list):
            return res
        else:
            return tuple(res)
    elif isinstance(signature, dict):
        res = {}
        for out_name, out_value in signature.items():
            res[out_name] = repack_ov_result_to_tf_format(ov_out, signature[out_name], out_name)
        return res
    raise Exception("Unknown type in FW signature: {}".format(type(signature)))


def get_output_signature_from_keras_layer(model):
    try:
        from openvino.frontend.tensorflow.utils import trace_tf_model_if_needed
        traced_model = trace_tf_model_if_needed(model, None, None, None)
        return traced_model.structured_outputs
    except:
        return None


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


def retrieve_inputs_info_for_signature(input_signature):
    inputs_info = []
    for input_name, input_info in (input_signature.items() if isinstance(input_signature, dict) else input_signature):
        input_shape = []
        try:
            if input_info.shape.as_list() == [None, None, None, 3] and input_info.dtype == tf.float32:
                # image classification case, let us imitate an image
                # that helps to avoid compute output size issue
                input_shape = [1, 200, 200, 3]
            elif input_info.shape.as_list() == [None, None, None, None, 3] and input_info.dtype == tf.float32:
                input_shape = [1, 2, 100, 100, 3]
            else:
                for dim in input_info.shape.as_list():
                    if dim is None:
                        input_shape.append(1)
                    else:
                        input_shape.append(dim)
        except ValueError:
            # unknown rank case
            # assume only one dimension
            input_shape = [3]
            pass
        if input_info.dtype == tf.resource:
            # skip inputs corresponding to variables
            continue
        assert input_info.dtype in type_map, "Unsupported input type: {}".format(input_info.dtype)
        inputs_info.append((input_name, input_shape, type_map[input_info.dtype]))

    return inputs_info
