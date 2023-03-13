# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import logging as log
import numpy as np
from openvino.tools.mo.moc_frontend.shape_utils import get_static_shape, get_dynamic_dims, parse_input_shapes
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.arg_wrappers import InputCutInfo
from openvino.tools.mo.utils.cli_parser import parse_input_value, split_inputs
from openvino.runtime import Tensor

def get_onnx_temp_filename(output_dir):
    output_dir = output_dir if output_dir is not None else os.getcwd()
    return os.path.normpath(os.path.join(output_dir, "model.onnx"))


def remove_tmp_onnx_model(out_dir):
    if not os.environ.get('SAVE_TO_BYTES_IO_ONNX_MODEL'):
        tmp_onnx_model = get_onnx_temp_filename(out_dir)

        if os.path.exists(tmp_onnx_model):
            os.remove(tmp_onnx_model)


def get_pytorch_decoder(model, input_shape, example_inputs, args):
    import torch
    import inspect
    try:
        from openvino.frontend.pytorch.decoder import TorchScriptPythonDecoder
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    inputs = prepare_torch_inputs(example_inputs, input_shape, allow_none=True)
    model.eval()
    input_signature = None
    if isinstance(model, torch.nn.Module) and not isinstance(model, torch.jit._trace.TopLevelTracedModule):
        input_signature = list(inspect.signature(model.forward).parameters.keys())
        try:
            scripted = torch.jit.script(model)
        except Exception as scripting_err:
            if inputs is not None:
                try:
                    scripted = torch.jit.trace(model, inputs)
                except Exception as tracing_e:
                    log.error('Both traicing and scripting failed')
                    raise tracing_e
            else:
                log.error("Model scripting failed")
                raise scripting_err
    else:
        scripted = model
    f_model = torch.jit.freeze(scripted)
    decoder = TorchScriptPythonDecoder(f_model)
    input_signature = align_input_parameters(f_model, input_signature, inputs, input_shape, args)
        
    return decoder, input_signature

def align_input_parameters(f_model, input_signature, inputs, input_shape, args):
    import torch
    try:
        from openvino.frontend.pytorch.decoder import pt_to_ov_type_map
    except Exception as e:
        log.error("PyTorch frontend loading failed")
    input_names = [str(inp.unique()) for inp in f_model.inlined_graph.inputs()][1:]
    input_debug_names = [inp.debugName() for inp in f_model.inlined_graph.inputs()][1:]
    if inputs is not None:
        if isinstance(inputs, dict):
            if input_signature is not None:
                ordered_inputs = []
                used_sign = []
                for key in input_signature:
                    if key not in inputs:
                        continue
                    ordered_inputs.append(inputs[key])
                    used_sign.append(key)
                inputs = ordered_inputs
                input_signature = used_sign
            else:
                inputs = list(inputs.values())
                input_signature = input_signature[:len(inputs)]
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        input_info = []
        for idx, input_data in enumerate(inputs):
            input_name = input_names[idx]
            input_data_rank = input_data.ndim
            pt_dtype = input_data.dtype if isinstance(input_data, torch.Tensor) else type(input_data)
            dtype = pt_to_ov_type_map.get(str(pt_dtype))
            input_sh = input_shape[idx] if input_shape is not None else [-1] * input_data_rank
            input_info.append(InputCutInfo(input_name, input_sh, dtype, None))
        
        input_argv = args.get("input")
        if input_argv is None:
            if input_info:
                args["input"] = input_info
                args.pop("input_shape", None)
            return input_signature
        input_info_by_name = {
            inp.name: inp for inp in input_info
        }
        debug_name_to_input_name = {debug_name: inp_name for debug_name, inp_name in zip(input_debug_names, input_names)}
        user_input_info_by_name = {}
        input_sign_to_input_name = {
            input_s: input_idx for input_s, input_idx in 
            zip(input_signature if input_signature is not None else input_debug_names, input_names)
        }
        if isinstance(input_argv, str):
            input_argv = [parse_input_value(input_v) for input_v in split_inputs(input_argv)]
            
        for input_tuple in input_argv:
            user_input_name = input_tuple[0]
            if user_input_name not in input_names and user_input_name not in input_debug_names:
                input_name = input_sign_to_input_name.get(user_input_name)
                if input_name is None:
                    raise Error(f"Unknown input name: {user_input_name}")
                if input_name in debug_name_to_input_name:
                    input_name = debug_name_to_input_name[input_name]
                user_input_name = input_name
            shape = input_tuple[1]
            if shape is None and input_info_by_name:
                shape = input_info_by_name[user_input_name][1]
            dtype = input_tuple[2]
            if dtype is None and input_info_by_name:
                dtype = input_info_by_name[user_input_name][2]
                
            user_input_info_by_name[user_input_name] = InputCutInfo(user_input_name, shape, dtype, input_tuple[3])
        
        input_info_by_name.update(user_input_info_by_name)
        updated_input_args = list(input_info_by_name.values())
        args["input"] = updated_input_args
        # avoid twise shape assignment
        args.pop("input_shape", None)
    return input_signature


def to_torch_tensor(tensor):
    import torch
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    if isinstance(tensor, Tensor):
        return torch.tensor(tensor.data)
    else:
        raise Error("Unexpected type of example_input. Supported types torch.Tensor, np.array or ov.Tensor. "
                    "Got {}".format(type(tensor)))


def prepare_torch_inputs(example_inputs, input_shape, allow_none=False):
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
    elif input_shape is not None:
        inputs = []
        for shape in input_shape:
            static_shape = get_static_shape(shape, dynamic_value=1)
            inputs.append(torch.zeros(static_shape))
        inputs = tuple(inputs)
    else:
        if not allow_none:
            raise Error("Please provide input_shape or example_input for converting PyTorch model.")
    return inputs


def convert_pytorch_to_onnx(model, input_shape, opset_version, example_inputs, output_dir):
    import io
    import torch

    input_names = None
    inputs = prepare_torch_inputs(example_inputs, input_shape)

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
                                            opset_version,
                                            example_inputs,
                                            out_dir)

        args['input_model'] = model_onnx

        ov_model, argv = main_convert(cli_parser, framework, args)
    except Exception as e:
        raise e
    finally:
        remove_tmp_onnx_model(out_dir)
    return ov_model, argv


def resolve_input_signature(argv, ov_model):
    def add_tensor_name(input_desc, input_name):
        tensor = input_desc.get_tensor()
        input_names = tensor.names
        input_names.update(input_name)
        tensor.set_names(input_names)

    input_signature = getattr(argv, "input_signature", None)
    if input_signature is not None:
        for idx, input_tensor in enumerate(ov_model.inputs):
            add_tensor_name(input_tensor, input_signature[idx])
    return ov_model
