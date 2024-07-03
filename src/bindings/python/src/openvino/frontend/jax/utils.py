# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax
import numpy as np
import jax.numpy as jnp

from openvino.runtime import op, Type as OVType, Shape, OVAny

numpy_to_ov_type_map = {
    np.float32: OVType.f32,
    bool: OVType.boolean,
    jax.dtypes.bfloat16: OVType.bf16, # TODO: check this
    np.float16: OVType.f16,
    np.float32: OVType.f32,
    np.float64: OVType.f64,
    np.uint8: OVType.u8,
    np.int8: OVType.i8,
    np.int16: OVType.i16,
    np.int32: OVType.i32,
    np.int64: OVType.i64,
}

jax_to_ov_type_map = {
    jnp.float32: OVType.f32,
    jnp.bfloat16: OVType.bf16, # TODO: check this
    jnp.float16: OVType.f16,
    jnp.float64: OVType.f64,
    jnp.uint8: OVType.u8,
    jnp.int8: OVType.i8,
    jnp.int16: OVType.i16,
    jnp.int32: OVType.i32,
    jnp.int64: OVType.i64,
}

try:
    jax_to_ov_type_map[jnp.bool] = OVType.boolean
except:
    pass

basic_to_ov_type_map = {
    int: OVType.i64,
    float: OVType.f32,
    bool: OVType.boolean,
}

ov_type_to_int_map = {
    OVType.u8: 0,
    OVType.i8: 1,
    OVType.i16: 2,
    OVType.i32: 3,
    OVType.i64: 4,
    OVType.f16: 5,
    OVType.f32: 6,
    OVType.f64: 7,
    OVType.boolean: 11,
    OVType.bf16: 15,
}

def get_type_from_py_type(value):
    if isinstance(value, float):
        return OVType.f32
    if isinstance(value, bool):
        return OVType.boolean
    if isinstance(value, int):
        return OVType.i64
    return OVType.dynamic

def get_ov_type_for_value(value):
    if isinstance(value, (jax.core.Var, jax.core.Literal)):
        if value.aval.dtype in jax_to_ov_type_map:
            return OVAny(jax_to_ov_type_map[value.aval.dtype])
        for k, v in numpy_to_ov_type_map.items():
            if value.aval.dtype == k:
                return OVAny(v)
        for k, v in basic_to_ov_type_map.items():
            if isinstance(value.aval.dtype, k):
                return OVAny(v)
    elif isinstance(value, (int, float, bool)):
        return OVAny(jax_to_ov_type_map[type(value)])
    else:
        raise NotImplementedError(f"dtype for {value} of type {type(value)} has not been supported yet.")
    
def get_ov_type_from_jax_type(dtype):
    if dtype in jax_to_ov_type_map:
        return OVAny(jax_to_ov_type_map[dtype])
    for k, v in numpy_to_ov_type_map.items():
        if dtype == k:
            return OVAny(v)
    for k, v in basic_to_ov_type_map.items():
        if isinstance(dtype, k):
            return OVAny(v)
    return None

def jax_array_to_ov_const(arr: np.ndarray, shared_memory=True):
    # TODO: deal with bfloat16 dtype here.
    if isinstance(arr, np.ndarray):
        return op.Constant(arr, shared_memory=shared_memory)
    elif isinstance(arr, jax.Array):
        return op.Constant(np.array(jax.device_get(arr)), shared_memory=shared_memory)
    else:
        raise ValueError(f"Constant is expected to be a numpy array or jax array but got {type(arr)}")

def ivalue_to_constant(ivalue, shared_memory=True):
    '''
    Convert a python object to an openvino constant.
    '''
    ov_type = get_type_from_py_type(ivalue)
    if ov_type.is_static():
        return op.Constant(ov_type, Shape([]), [ivalue]).outputs()

    if isinstance(ivalue, (list, tuple)):
        assert len(ivalue) > 0, "Can't deduce type for empty list"
        ov_type = get_type_from_py_type(ivalue[0])
        assert ov_type.is_static(), "Can't deduce type for list"
        return op.Constant(ov_type, Shape([len(ivalue)]), ivalue).outputs()

    if isinstance(ivalue, (jax.Array, np.ndarray)):
        return jax_array_to_ov_const(ivalue, shared_memory=shared_memory).outputs()
            
    ov_dtype_value = get_ov_type_from_jax_type(ivalue)
    if ov_dtype_value is not None:
        return op.Constant(OVType.i64, Shape([]), [ov_type_to_int_map[ov_dtype_value]]).outputs()
    
    print(f"[WARNING][JAX FE] Cannot get constant from value {ivalue}")
    return None