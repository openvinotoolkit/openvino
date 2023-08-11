
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import torch
import numpy as np
import ctypes

from openvino.runtime import op, Type as OVType, Shape, Tensor


def maybe_convert_max_int(value : int):
    # FIXME: This is a convertion from 64-bit positive max integer value
    # to 32-bit positive max integer value. Find a better way to handle this.
    if value == torch.iinfo(torch.int64).max:
        return torch.iinfo(torch.int32).max
    else:
        return value

def make_constant(*args, **kwargs):
    return op.Constant(*args, **kwargs)

def fetch_attr(self_module, target : str):
    """
    Fetch an attribute from the ``Module`` hierarchy of ``self.module``.

    Args:
        target (str): The fully-qualified name of the attribute to fetch

    Return:
        Any: The value of the attribute.
    """
    target_atoms = target.split('.')
    attr_itr = self_module
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def get_type_from_py_type(value):
    if isinstance(value, float):
        return OVType.f32
    if isinstance(value, bool):
        return OVType.boolean
    if isinstance(value, int):
        # Python int is 64 bit, but we will convert it to int32 except cases when it can't fit in 32 bits
        if torch.iinfo(torch.int).min <= value <= torch.iinfo(torch.int).max:
            return OVType.i32
        return OVType.i64
    return OVType.dynamic


def torch_tensor_to_ov_const(torch_t: torch.Tensor, shared_memory=True):
    torch_t = torch_t.contiguous()
    if torch_t.dtype == torch.bfloat16:
        # reinterpret bfloat16 data as float16 to allow conversion to numpy
        torch_t = torch_t.view(torch.float16)
        narr = torch_t.numpy(force=True)
        tensor = Tensor(narr, torch_t.shape, OVType.bf16)
        ov_const = op.Constant(tensor, shared_memory=shared_memory)
    else:
        narr = torch_t.numpy(force=True)
        ov_const = op.Constant(narr, shared_memory=shared_memory)
    return ov_const


def ivalue_to_constant(ivalue, shared_memory=True):
    ov_type = get_type_from_py_type(ivalue)
    if ov_type.is_static():
        return op.Constant(ov_type, Shape([]), [ivalue]).outputs()

    if isinstance(ivalue, (list, tuple)):
        assert len(ivalue) > 0, "Can't deduce type for empty list"
        ov_type = get_type_from_py_type(ivalue[0])
        assert ov_type.is_static(), "Can't deduce type for list"
        return op.Constant(ov_type, Shape([len(ivalue)]), ivalue).outputs()

    if isinstance(ivalue, torch.Tensor):
        return torch_tensor_to_ov_const(ivalue, shared_memory=shared_memory).outputs()
    return None

def get_value_from_getattr(getattr_node, self_module):
    assert getattr_node.kind() == "prim::GetAttr", "Got node of kind not equal to prim::GetAttr"
    # GetAttr nodes can be nested
    stack = []
    while getattr_node.kind() == "prim::GetAttr":
        stack.append(getattr_node)
        inputs = list(getattr_node.inputs())
        if len(inputs) == 0:
            break
        getattr_node = inputs[0].node()
    module = self_module
    while len(stack) > 0:
        node = stack.pop()
        attr_name = node.s("name")
        assert hasattr(module, attr_name), f"No attribute with name \"{attr_name}\" found in module."
        module = getattr(module, attr_name)
    return module

pt_to_ov_type_map = {
    "float": OVType.f32,
    "int": OVType.i32,
    "bool": OVType.boolean,
    "torch.bfloat16": OVType.bf16,
    "torch.float16": OVType.f16,
    "torch.float32": OVType.f32,
    "torch.float64": OVType.f64,
    "torch.uint8": OVType.u8,
    "torch.int8": OVType.i8,
    "torch.int32": OVType.i32,
    "torch.int64": OVType.i64,
    "torch.bool": OVType.boolean,
    "torch.DoubleTensor": OVType.f64,
    "torch.FloatTensor": OVType.f32,
    "torch.IntTensor": OVType.i32,
    "torch.LongTensor": OVType.i64,
    "torch.BoolTensor": OVType.boolean,
    "torch.quint8": OVType.u8,
    "torch.qint8": OVType.i8,
    "torch.qint32": OVType.i32
}

ov_to_c_type_map = {
    OVType.f32: ctypes.c_float,
    OVType.f64: ctypes.c_double,
    OVType.i32: ctypes.c_int,
    OVType.i64: ctypes.c_int64,
}
