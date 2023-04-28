# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import logging as log
import numpy as np
from openvino.tools.mo.moc_frontend.shape_utils import get_static_shape, get_dynamic_dims, parse_input_shapes
from openvino.tools.mo.utils.error import Error
from openvino.runtime import Tensor, Type
from openvino.runtime.utils.types import get_element_type_str
from openvino.tools.mo.utils.cli_parser import input_to_input_cut_info, input_shape_to_input_cut_info

def get_onnx_temp_filename(output_dir):
    output_dir = output_dir if output_dir is not None else os.getcwd()
    return os.path.normpath(os.path.join(output_dir, "model.onnx"))


def remove_tmp_onnx_model(out_dir):
    if not os.environ.get('SAVE_TO_BYTES_IO_ONNX_MODEL'):
        tmp_onnx_model = get_onnx_temp_filename(out_dir)

        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)


def get_pytorch_decoder(model, input_shape, example_inputs, input_info):
    try:
        from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    inputs = prepare_torch_inputs(example_inputs, input_shape, input_info, allow_none=True)
    decoder = TorchScriptPythonDecoder(model, example_input=inputs)
        
    return decoder


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
        "i16":torch.int16,
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
    raise Error(f"Unexpected data type for input. Supporteed torch.dtype, numpy.dtype, ov.Type and str. Got {type(dtype)}")


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
        for inp in input_info:
            shape = inp.shape
            if shape is None:
                if not allow_none:
                    raise Error("Please provide input_shape or example_input for all inputs converting PyTorch model.")
                inputs = None
                break
            dtype = get_torch_dtype(inp.type)
            static_shape = get_static_shape(shape, dynamic_value=1)
            inputs.append(torch.zeros(static_shape, dtype=dtype))
        if isinstance(inputs, list):
            inputs = tuple(inputs)
    else:
        if not allow_none:
            raise Error("Please provide input_shape or example_input for converting PyTorch model.")
    return inputs


def convert_pytorch_to_onnx(model, input_shape, input_info, opset_version, example_inputs, output_dir):
    import io
    import torch

    input_names = None
    inputs = prepare_torch_inputs(example_inputs, input_shape, input_info)

    dynamic_dims_dict = {}
    if input_shape is not None and input_names is None:
        input_names = ["input_{}".format(idx) for idx in range(len(input_shape))]
        for shape_idx, shape in enumerate(input_shape):
            dynamic_dims = get_dynamic_dims(shape)
            if len(dynamic_dims) > 0:
                dynamic_dims_dict[input_names[shape_idx]] = dynamic_dims
    additional_params = {}
    if len(dynamic_dims_dict) > 0:
        additional_params.update({'dynamic_axes': dynamic_dims_dict})
    if input_names is not None and len(input_names) > 0:
        additional_params.update({'input_names': input_names})

    if os.environ.get('SAVE_TO_BYTES_IO_ONNX_MODEL'):
        model_onnx = io.BytesIO()
    else:
        model_onnx = get_onnx_temp_filename(output_dir)
    if opset_version is not None:
        additional_params.update({'opset_version': opset_version})

    torch.onnx.export(model,
                      inputs,
                      model_onnx,
                      operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                      **additional_params)
    return model_onnx


def convert_pytorch_via_onnx(args, example_inputs, cli_parser, framework, main_convert):
    opset_version = None
    if 'onnx_opset_version' in args and args['onnx_opset_version'] is not None:
        opset_version = args['onnx_opset_version']
    out_dir = args['output_dir'] if 'output_dir' in args else None
    if os.environ.get('SAVE_TO_BYTES_IO_ONNX_MODEL'):
        args['use_legacy_frontend'] = True
    # these parameters used only on PyTorch to ONNX conversion, 
    # remove them before passing model to next step
    args['example_input'] = None
    args['onnx_opset_version'] = None
    try:
        model_onnx = convert_pytorch_to_onnx(args['input_model'],
                                            parse_input_shapes(args),
                                            args.get("input"),
                                            opset_version,
                                            example_inputs,
                                            out_dir)

        args['input_model'] = model_onnx

        ov_model, argv = main_convert(cli_parser, framework, args, True)
    except Exception as e:
        raise e
    finally:
        remove_tmp_onnx_model(out_dir)
    return ov_model, argv
