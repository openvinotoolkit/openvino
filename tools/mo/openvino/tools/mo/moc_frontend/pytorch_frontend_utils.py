# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np
from openvino.tools.mo.moc_frontend.shape_utils import get_static_shape
from openvino.tools.mo.utils.error import Error
from openvino.runtime import Tensor, Type, PartialShape
from openvino.runtime.utils.types import get_element_type_str
from openvino.tools.mo.utils.cli_parser import input_to_input_cut_info, input_shape_to_input_cut_info


def get_pytorch_decoder(model, input_shape, example_inputs, args):
    try:
        from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    inputs = prepare_torch_inputs(example_inputs, input_shape, args.get("input"), allow_none=True)
    decoder = TorchScriptPythonDecoder(model, example_input=inputs)
    args['input_model'] = decoder
    args["framework"] = "pytorch"
    if inputs is not None:
        extract_input_info_from_example(inputs, args)

    return args


def extract_input_info_from_example(example_inputs, args):
    from openvino.tools.mo import InputCutInfo
    try:
        from openvino.frontend.pytorch.decoder import pt_to_ov_type_map
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    inputs = input_to_input_cut_info(args.get("input", []))
    input_shapes = args.get("input_shape")
    input_shape_to_input_cut_info(input_shapes, inputs)
    is_dict_input = isinstance(example_inputs, dict)
    list_inputs = list(example_inputs.values()) if is_dict_input else example_inputs
    input_names = None if not is_dict_input else list(example_inputs)
    if not isinstance(list_inputs, (list, tuple)):
        list_inputs = [list_inputs]
    updated_input_info = []
    if inputs:
        for input_id, input_info in enumerate(inputs):
            input_name = input_info.name
            if is_dict_input and input_name in example_inputs:
                example_input = example_inputs[input_name]
            else:
                example_input = list_inputs[input_id]
                if is_dict_input and input_name is None:
                    input_name = input_names[input_id]
            dtype = getattr(example_input, "dtype", type(example_input))
            ov_dtype = pt_to_ov_type_map.get(str(dtype)) if not input_info.type else input_info.type
            data_rank = getattr(example_input, "ndim", 0)
            input_shape = input_info.shape if input_info.shape is not None else PartialShape([-1] * data_rank)

            updated_input_info.append(
                InputCutInfo(input_info.name, input_shape, ov_dtype, input_info.value)
            )
    else:
        for input_id, example_input in enumerate(list_inputs):
            dtype = getattr(example_input, "dtype", type(example_input))
            ov_dtype = pt_to_ov_type_map.get(str(dtype))
            data_rank = getattr(example_input, "ndim", 0)
            input_shape =  PartialShape([-1] * data_rank)
            input_name = input_names[input_id] if input_names else None
            updated_input_info.append(
                InputCutInfo(input_name, input_shape, ov_dtype, None)
            )
    args.pop("input_shape")
    args["input"] = updated_input_info

def to_torch_tensor(tensor):
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    if isinstance(tensor, Tensor):
        return torch.tensor(tensor.data)
    if isinstance(tensor, (float, int, bool)):
        return tensor
    else:
        raise Error("Unexpected type of example_input. Supported types torch.Tensor, np.array or ov.Tensor. "
                    "Got {}".format(type(tensor)))


def get_torch_dtype(dtype):
    import torch
    ov_str_to_torch = {
        "boolean": torch.bool,
        "f16": torch.float16,
        "f32": torch.float32,
        "f64": torch.float64,
        "i8": torch.int8,
        "i16": torch.int16,
        "i32": torch.int32,
        "i64": torch.int64,
        "u8": torch.uint8,
    }
    if dtype is None:
        return torch.float
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, (type, np.dtype)):
        dtype = get_element_type_str(dtype)
    if isinstance(dtype, Type):
        dtype = dtype.get_type_name()
    if isinstance(dtype, str):
        str_dtype = ov_str_to_torch.get(dtype)
        if str_dtype is None:
            raise Error(f"Unexpected data type '{dtype}' for input")
        return str_dtype
    raise Error(f"Unexpected data type for input. Supported torch.dtype, numpy.dtype, ov.Type and str. Got {type(dtype)}")


def prepare_torch_inputs(example_inputs, input_shape, input_info=None, allow_none=False):
    import torch
    inputs = None
    if example_inputs is not None:
        inputs = example_inputs
        if isinstance(inputs, list):
            inputs = [to_torch_tensor(x) for x in inputs]
            if len(inputs) == 1:
                inputs = torch.unsqueeze(inputs[0], 0)
            else:
                inputs = inputs
        elif isinstance(inputs, tuple):
            inputs = [to_torch_tensor(x) for x in inputs]
            inputs = tuple(inputs)
        elif isinstance(inputs, dict):
            for name, tensor in inputs.items():
                assert isinstance(name, str), "Expected dictionary where keys are input names of string type and" \
                                              " values are tensors. Got key of type {}".format(type(name))
                inputs[name] = to_torch_tensor(tensor)
        else:
            inputs = to_torch_tensor(inputs)
    elif input_info is not None or input_shape is not None:
        input_info = input_to_input_cut_info(input_info) or []
        input_shape_to_input_cut_info(input_shape, input_info)
        inputs = []
        inputs_with_names = {}
        for inp in input_info:
            shape = inp.shape
            if shape is None:
                if not allow_none:
                    raise Error("Please provide input_shape or example_input for all inputs converting PyTorch model.")
                inputs = None
                break
            dtype = get_torch_dtype(inp.type)
            static_shape = get_static_shape(shape, dynamic_value=1)
            input_tensor = torch.zeros(static_shape, dtype=dtype)
            if inp.name is not None:
                inputs_with_names[inp.name] = input_tensor
            inputs.append(input_tensor)
        if isinstance(inputs, list):
            inputs = tuple(inputs)
        if inputs is not None and len(inputs) == len(inputs_with_names):
            inputs = inputs_with_names
    else:
        if not allow_none:
            raise Error("Please provide input_shape or example_input for converting PyTorch model.")
    return inputs
