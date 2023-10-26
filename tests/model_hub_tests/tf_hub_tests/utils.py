# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import tensorflow as tf

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
