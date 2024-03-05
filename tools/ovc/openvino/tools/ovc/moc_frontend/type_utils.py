# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys

import openvino as ov
from openvino.runtime import Type


def is_type(val):
    if isinstance(val, (type, Type)):
        return True
    if 'tensorflow' in sys.modules:
        import tensorflow as tf # pylint: disable=import-error
        if isinstance(val, tf.dtypes.DType):
            return True
    if 'torch' in sys.modules:
        import torch
        if isinstance(val, torch.dtype):
            return True
    if 'paddle' in sys.modules:
        import paddle
        if isinstance(val, paddle.dtype):
            return True
    return False


def to_ov_type(val):
    if isinstance(val, Type):
        return val
    if isinstance(val, type):
        return Type(val)
    if 'tensorflow' in sys.modules:
        import tensorflow as tf # pylint: disable=import-error
        if isinstance(val, tf.dtypes.DType):
            return Type(val.as_numpy_dtype())
    if 'torch' in sys.modules:
        import torch

        if isinstance(val, torch.dtype):
            torch_to_ov_type = {
                torch.float32: ov.Type.f32,
                torch.float16: ov.Type.f16,
                torch.float64: ov.Type.f64,
                torch.bfloat16: ov.Type.bf16,
                torch.uint8: ov.Type.u8,
                torch.int8: ov.Type.i8,
                torch.int16: ov.Type.i16,
                torch.int32: ov.Type.i32,
                torch.int64: ov.Type.i64,
                torch.bool: ov.Type.boolean,
            }
            if val not in torch_to_ov_type:
                raise Exception("The provided data time is not supported {}.".format(val))

            return torch_to_ov_type[val]

    if 'paddle' in sys.modules:
        import paddle

        if isinstance(val, paddle.dtype):
            paddle_to_ov_type = {
                paddle.float32: ov.Type.f32,
                paddle.float16: ov.Type.f16,
                paddle.float64: ov.Type.f64,
                paddle.bfloat16: ov.Type.bf16,
                paddle.uint8: ov.Type.u8,
                paddle.int8: ov.Type.i8,
                paddle.int16: ov.Type.i16,
                paddle.int32: ov.Type.i32,
                paddle.int64: ov.Type.i64,
                paddle.bool: ov.Type.boolean,
            }

            if val not in paddle_to_ov_type:
                raise Exception("The provided data time is not supported {}.".format(val))

            return paddle_to_ov_type[val]
    raise Exception("Unexpected type object. Expected ov.Type, np.dtype, tf.dtypes.DType. Got {}".format(type(val)))
