# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf

rng = np.random.default_rng(seed=56190)

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


def prepare_input(input_shape, input_type):
    if input_type in [np.float32, np.float64]:
        return 2.0 * rng.random(size=input_shape, dtype=input_type)
    elif input_type in [np.uint8, np.uint16, np.int8, np.int16, np.int32, np.int64]:
        return rng.integers(0, 5, size=input_shape).astype(input_type)
    elif input_type in [str]:
        return np.broadcast_to("Some string", input_shape)
    elif input_type in [bool]:
        return rng.integers(0, 2, size=input_shape).astype(input_type)
    else:
        assert False, "Unsupported type {}".format(input_type)


def prepare_inputs(inputs_info):
    # if len(inputs_info) > 0 and inputs_info[0] == 'list':
    #     inputs = []
    #     inputs_info = inputs_info[1:]
    #     for input_name, input_shape, input_type in inputs_info:
    #         inputs.append(prepare_input(input_shape, input_type))
    # else:
    inputs = {}
    for input_name, input_shape, input_type in inputs_info:
        inputs[input_name] = prepare_input(input_shape, input_type)
    return inputs


def get_inputs_info(model_obj):
    inputs_info = []
    assert len(model_obj.structured_input_signature) > 1, "incorrect model or test issue"
    for input_name, input_info in model_obj.structured_input_signature[1].items():
        input_shape = []
        try:
            if input_info.shape.as_list() == [None, None, None, 3] and input_info.dtype == tf.float32:
                # image classification case, let us imitate an image
                # that helps to avoid compute output size issue
                input_shape = [1, 200, 200, 3]
            else:
                for dim in input_info.shape.as_list():
                    if dim is None:
                        input_shape.append(1)
                    else:
                        input_shape.append(dim)
        except ValueError:
            # unknown rank case
            pass
        if input_info.dtype == tf.resource:
            # skip inputs corresponding to variables
            continue
        assert input_info.dtype in type_map, "Unsupported input type: {}".format(input_info.dtype)
        inputs_info.append((input_name, input_shape, type_map[input_info.dtype]))

    return inputs_info


def generate_tf_hub_inputs(model):
    """
    Generates random inputs depending on model's input type
    """
    return prepare_inputs(get_inputs_info(model))
