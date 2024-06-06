# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import jax
import numpy as np

from openvino.runtime import op, Type as OVType, Shape, Tensor
from openvino.runtime import opset11 as ops

jax_to_ov_type_map = {
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
    
    int: OVType.i64,
    float: OVType.f32,
    bool: OVType.boolean,
}