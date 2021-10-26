# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.impl import Type

def pack_data(array: np.ndarray, ov_type: Type) -> np.ndarray:
    assert ov_type in [Type.u1, Type.u4, Type.i4]

    minimum_regular_dtype = np.int8 if ov_type == Type.i4 else np.uint8
    casted_to_regular_type = array.astype(dtype=minimum_regular_dtype, casting="unsafe")
    if not np.array_equal(casted_to_regular_type, array):
        raise RuntimeError(f'The conversion of array "{array}" to dtype "{casted_to_regular_type}" results in rounding')

    data_size = casted_to_regular_type.size
    num_bits = ov_type.bitwidth

    assert num_bits < 8 and 8 % num_bits == 0, "Packing algorithm for the data types stored in 1, 2 or 4 bits"
    num_values_fitting_into_uint8 = 8 // num_bits
    pad = (-data_size) % num_values_fitting_into_uint8

    flattened = casted_to_regular_type.flatten()
    padded = np.concatenate((flattened, np.zeros([pad], dtype=minimum_regular_dtype)))
    assert padded.size % num_values_fitting_into_uint8 == 0

    bit_order_little = (padded[:, None] & (1 << np.arange(num_bits)) > 0).astype(np.uint8)
    bit_order_big = np.flip(bit_order_little, axis=1)
    bit_order_big_flattened = bit_order_big.flatten()

    return np.packbits(bit_order_big_flattened)