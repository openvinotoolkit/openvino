# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa

import numpy as np
from typing import Union
from openvino import Type, Shape


def pack_data(array: np.ndarray, type: Type) -> np.ndarray:
    """Represent array values as u1, u2, u3, u4, u6, i4, nf4 or f4e2m1 element type and pack them into uint8 numpy array.

    For u1: Bit packing from the most significant bit, like the core
    For u2, u4, i4, nf4, f4e2m1: Bit packing from the least significant bits, like the core
    For u3, u6: Linear LSB-first bit-stream packing, values may straddle a byte boundary

    For the byte-aligned types (u1, u2, u4, i4, nf4, f4e2m1) the values are padded with zeros up to a
    multiple of the number of values that fit into one byte (8 // bitwidth).

    Example: two uint8 values - [7, 8] can be represented as uint4 values and be packed into one uint8
             value - [135]. The first value takes the low 4 bits and the second the high 4 bits,
             so [0111, 1000] is viewed as [10000111], which is bit representation of [135].

    :param array: numpy array with values to pack.
    :type array: numpy array
    :param type: Type to interpret the array values. Type must be u1, u2, u3, u4, u6, i4, nf4 or f4e2m1.
    :type type: openvino.Type
    :raises RuntimeError: if a value doesn't fit the type: [0, 2^bitwidth - 1] for unsigned types,
                          [-8, 7] for i4, 4-bit codes [0, 15] for nf4 and f4e2m1. u1 is not checked.
    """
    _check_range(array, type)

    # Handle u3 and u6 with linear LSB-first bit-stream packing
    if type == Type.u3:
        return _pack_linear(array, 3)
    elif type == Type.u6:
        return _pack_linear(array, 6)
    
    assert type in [Type.u1, Type.u2, Type.u4, Type.i4, Type.nf4, Type.f4e2m1], "Packing algorithm for the" "data types stored in 1, 2 or 4 bits"

    minimum_regular_dtype = np.int8 if type == Type.i4 else np.uint8
    casted_to_regular_type = array.astype(dtype=minimum_regular_dtype, casting="unsafe")
    if not np.array_equal(casted_to_regular_type, array):
        raise RuntimeError(f'The conversion of array "{array}" to dtype' f' "{casted_to_regular_type}" results in rounding')

    data_size = casted_to_regular_type.size
    num_bits = type.bitwidth

    assert num_bits < 8 and 8 % num_bits == 0, "Packing algorithm for the" "data types stored in 1, 2 or 4 bits"
    num_values_fitting_into_uint8 = 8 // num_bits
    pad = (-data_size) % num_values_fitting_into_uint8

    flattened = casted_to_regular_type.flatten()
    padded = np.concatenate((flattened, np.zeros([pad], dtype=minimum_regular_dtype)))  # type: ignore
    assert padded.size % num_values_fitting_into_uint8 == 0

    bit_order_little = (padded[:, None] & (1 << np.arange(num_bits)) > 0).astype(minimum_regular_dtype)

    # Same layout as the core: u1 is packed from the most significant bit,
    # u2, u4, i4, nf4 and f4e2m1 from the least significant bits.
    if type == Type.u1:
        return np.packbits(bit_order_little.flatten(), bitorder="big")
    return np.packbits(bit_order_little.flatten(), bitorder="little")


def unpack_data(array: np.ndarray, type: Type, shape: Union[list, Shape]) -> np.ndarray:
    """Extract openvino element type values from array into new uint8/int8 array given shape.

    For u1: Bit unpacking from the most significant bit, like the core
    For u2, u4, i4, nf4, f4e2m1: Bit unpacking from the least significant bits, like the core
    For u3, u6: Linear LSB-first bit-stream unpacking, values may straddle a byte boundary

    Example: uint8 value [135] can be represented as two u4 values and be unpacked into [7, 8]
             because [135] bit representation is [10000111]: the low 4 bits [0111] are the first
             value and the high 4 bits [1000] the second.

    :param array: numpy array to unpack.
    :type array: numpy array
    :param type: Type to extract from array values. Type must be u1, u2, u3, u4, u6, i4, nf4 or f4e2m1.
    :type type: openvino.Type
    :param shape: the new shape for the unpacked array.
    :type shape: Union[list, openvino.Shape]
    """
    # Handle u3 and u6 with linear LSB-first bit-stream unpacking
    if type == Type.u3:
        return _unpack_linear(array, 3, shape)
    elif type == Type.u6:
        return _unpack_linear(array, 6, shape)
    
    assert type in [Type.u1, Type.u2, Type.u4, Type.i4, Type.nf4, Type.f4e2m1], "Unpacking algorithm for the" "data types stored in 1, 2 or 4 bits"
    if type == Type.u1:
        return np.resize(np.unpackbits(array.view(np.uint8)), list(shape))

    unpacked = _unpack_linear(array, type.bitwidth, shape)
    if type == Type.i4:
        # Sign-extend: codes 8..15 are -8..-1
        return (unpacked.astype(np.int8) << 4) >> 4
    return unpacked


def _check_range(array: np.ndarray, type: Type) -> None:
    """Raise if a value doesn't fit the target type, as ops.constant does for the integer types and nf4.

    nf4 and f4e2m1 values are 4-bit codes. u1 is left out: the core stores any non-zero value as 1.
    """
    if type not in [Type.u2, Type.u3, Type.u4, Type.u6, Type.i4, Type.nf4, Type.f4e2m1]:
        return
    if type == Type.i4:
        low, high = -8, 7
    else:
        low, high = 0, (1 << type.bitwidth) - 1
    # Compare elementwise: np.min/np.max return NaN when any value is NaN, which would hide out-of-range values.
    # NaN itself is left to the existing checks, as before.
    if np.any((array < low) | (array > high)):
        min_value, max_value = np.nanmin(array), np.nanmax(array)
        raise RuntimeError(
            f"The values of the {array.dtype} array with shape {array.shape} (min {min_value}, max {max_value}) "
            f"are out of the {type.get_type_name()} range [{low}, {high}]"
        )


def _pack_linear(array: np.ndarray, bits: int) -> np.ndarray:
    """Pack values into a linear, LSB-first bit-stream (used for u3/u6): value i occupies bits
    [i * bits, i * bits + bits), possibly straddling a byte boundary.
    """
    array = array.astype(np.uint8, casting="unsafe").flatten()
    bit_order_little = (array[:, None] & (1 << np.arange(bits)) > 0).astype(np.uint8)
    return np.packbits(bit_order_little.flatten(), bitorder="little")


def _unpack_linear(array: np.ndarray, bits: int, shape: Union[list, Shape]) -> np.ndarray:
    """Unpack values from a linear, LSB-first bit-stream (used for every type except u1), inverse of _pack_linear."""
    array = array.view(np.uint8)
    shape = list(shape)
    num_values = int(np.prod(shape))

    bits_unpacked = np.unpackbits(array, bitorder="little")[: num_values * bits]
    values = bits_unpacked.reshape(-1, bits)
    weights = (1 << np.arange(bits)).astype(np.uint8)
    result = (values * weights).sum(axis=1).astype(np.uint8)
    return np.resize(result, shape)
