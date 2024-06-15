# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax
import numpy as np
import jax.numpy as jnp

from openvino.runtime import op, Type as OVType

def jax_array_to_ov_const(arr: np.ndarray, shared_memory=True):
    # TODO: deal with bfloat16 dtype here.
    if isinstance(arr, np.ndarray):
        return op.Constant(arr, shared_memory=shared_memory)
    elif isinstance(arr, jax.Array):
        return op.Constant(np.array(jax.device_get(arr)), shared_memory=shared_memory)
    else:
        raise ValueError(f"Constant is expected to be a numpy array or jax array but got {type(arr)}")

numpy_to_ov_type_map = {
    np.dtypes.Float32DType: OVType.f32,
    np.dtypes.BoolDType: OVType.boolean,
    jax.dtypes.bfloat16: OVType.bf16, # TODO: check this
    np.dtypes.Float16DType: OVType.f16,
    np.dtypes.Float32DType: OVType.f32,
    np.dtypes.Float64DType: OVType.f64,
    np.dtypes.UInt8DType: OVType.u8,
    np.dtypes.Int8DType: OVType.i8,
    np.dtypes.Int16DType: OVType.i16,
    np.dtypes.Int32DType: OVType.i32,
    np.dtypes.Int64DType: OVType.i64,
}

jax_to_ov_type_map = {
    jnp.float32: OVType.f32,
    jnp.bool: OVType.boolean,
    jnp.bfloat16: OVType.bf16, # TODO: check this
    jnp.float16: OVType.f16,
    jnp.float64: OVType.f64,
    jnp.uint8: OVType.u8,
    jnp.int8: OVType.i8,
    jnp.int16: OVType.i16,
    jnp.int32: OVType.i32,
    jnp.int64: OVType.i64,
}

basic_to_ov_type_map = {
    int: OVType.i64,
    float: OVType.f32,
    bool: OVType.boolean,
}