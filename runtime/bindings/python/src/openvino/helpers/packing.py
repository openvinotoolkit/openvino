# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from typing import Union
from openvino.impl import Type, Shape

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

    bit_order_little = (padded[:, None] & (1 << np.arange(num_bits)) > 0).astype(minimum_regular_dtype)
    bit_order_big = np.flip(bit_order_little, axis=1)
    bit_order_big_flattened = bit_order_big.flatten()

    return np.packbits(bit_order_big_flattened)


def unpack_data(array: np.ndarray, ov_type: Type, shape: Union[list, Shape]) -> np.ndarray:
    unpacked = np.unpackbits(array.view(np.uint8))
    shape = list(shape)
    if ov_type.bitwidth == 1:
        return np.resize(unpacked, shape)
    else:
        unpacked = unpacked.reshape(-1, ov_type.bitwidth)
        padding_shape = (unpacked.shape[0], 8 - ov_type.bitwidth)
        padding = np.ndarray(padding_shape, np.uint8)
        if ov_type == Type.i4:
            for axis, bits in enumerate(unpacked):
                if bits[0] == 1:
                    padding[axis] = np.ones((padding_shape[1],), np.uint8)
                else:
                    padding[axis] = np.zeros((padding_shape[1],), np.uint8)
        else:
            padding = np.zeros(padding_shape, np.uint8)
        padded = np.concatenate((padding, unpacked), 1)
        packed = np.packbits(padded, 1)
        if ov_type == Type.i4:
            return np.resize(packed, shape).astype(dtype=np.int8)
        else:
            return np.resize(packed, shape)
