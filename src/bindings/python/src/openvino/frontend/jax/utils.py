# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax
import numpy as np

from openvino.runtime import op, Type as OVType, Shape, Tensor
from openvino.runtime import opset11 as ops





jax_to_ov_type_map = {
    np.float32: OVType.f32,
    np.bool_: OVType.boolean,
    jax.dtypes.bfloat16: OVType.bf16, # TODO: check this
    np.float16: OVType.f16,
    np.float32: OVType.f32,
    np.float64: OVType.f64,
    np.uint8: OVType.u8,
    np.int8: OVType.i8,
    np.int16: OVType.i16,
    np.int32: OVType.i32,
    np.int64: OVType.i64,
    
    int: OVType.i64,
    float: OVType.f32,
    bool: OVType.boolean,
}