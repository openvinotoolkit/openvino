# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys

import numpy as np
# pylint: disable=no-name-in-module,import-error
from openvino.runtime import Tensor, PartialShape
from openvino.tools.ovc.error import Error



def get_pytorch_decoder(model, example_inputs, args):
    try:
        from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
        from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
        from openvino.frontend.pytorch.module_extension import ModuleExtension
        import torch
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    
    def extract_module_extensions(args):
        extensions = args.get('extension', []) or []
        if not isinstance(extensions, (list, tuple)):
            extensions = [extensions]
        return {extension.module: extension for extension in extensions if isinstance(extension, ModuleExtension)}

    if 'nncf' in sys.modules:
        is_good_version = True
        try:
            from nncf.torch.nncf_network import NNCFNetwork

            if isinstance(model, NNCFNetwork):
                from packaging import version
                if version.parse(sys.modules['nncf'].__version__) < version.parse("2.6"):
                    is_good_version = False
        except:
            pass
        if not is_good_version:
            raise RuntimeError(
                "NNCF models produced by nncf<2.6 are not supported directly. Please upgrade nncf or export to ONNX first.")
    inputs = prepare_torch_inputs(example_inputs)
    if not isinstance(model, (TorchScriptPythonDecoder, TorchFXPythonDecoder)):
        if hasattr(torch, "export") and isinstance(model, (torch.export.ExportedProgram)):
            from packaging import version
            if version.parse(torch.__version__) >= version.parse("2.2"):
                model = model.run_decompositions()
            gm = model.module()
            decoder = TorchFXPythonDecoder(gm)
        else:
            decoder = TorchScriptPythonDecoder(
                model,
                example_input=inputs,
                shared_memory=args.get("share_weights", True),
                module_extensions=extract_module_extensions(args))
    else:
        decoder = model
    args['input_model'] = decoder
    args["example_input"] = inputs

    return args


def update_list_or_dict(container, name, idx, value):
    if isinstance(container, dict):
        if name is None:
            name = list(container)[idx]
        container[name] = value
        return
    if idx == len(container):
        container.append(value)
    elif idx > len(container):
        raise Error(f"Wrong {idx}")
    else:
        container[idx] = value
    return


def get_value_from_list_or_dict(container, name, idx):
    if isinstance(container, dict):
        if name is None:
            if idx < len(container):
                name = list(container)[idx]
            return None
        return container.get(name)
    if idx < len(container):
        return container[idx]
    return None


def extract_input_info_from_example(args, inputs):
    try:
        from openvino.frontend.pytorch.utils import pt_to_ov_type_map  # pylint: disable=no-name-in-module,import-error
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    example_inputs = args.example_input
    data_types = args.placeholder_data_types or {}
    input_shapes = args.placeholder_shapes or {}
    is_dict_input = isinstance(example_inputs, dict)
    list_inputs = list(example_inputs.values()) if is_dict_input else example_inputs
    input_names = None
    if not isinstance(example_inputs, (list, tuple, dict)):
        list_inputs = [list_inputs]
    if args.input_model._input_is_list:
        list_inputs[0] = list_inputs[0].unsqueeze(0)
    if args.input_model._input_signature is not None and not is_dict_input:
        input_names = args.input_model._input_signature[1:] if args.input_model._input_signature[
                                                                   0] == "self" else args.input_model._input_signature
        if not is_dict_input:
            example_inputs = dict(zip(input_names, list_inputs))
            is_dict_input = True
    elif is_dict_input:
        input_names = list(example_inputs)
    if not data_types and input_names is None:
        data_types = []
    if not input_shapes and input_names is None:
        input_shapes = []
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
            example_dtype = pt_to_ov_type_map.get(str(dtype))
            user_dtype = get_value_from_list_or_dict(data_types, input_name, input_id)
            if user_dtype is not None and example_dtype is not None and example_dtype != user_dtype:
                raise Error(
                    f"Defined input type {user_dtype} is not equal to provided example_input type {example_dtype}")

            data_rank = getattr(example_input, "ndim", 0)
            user_input_shape = get_value_from_list_or_dict(input_shapes, input_name, input_id)
            if user_input_shape.rank.is_static and user_input_shape.rank.get_length() != data_rank:
                raise Error(
                    f"Requested input shape {user_input_shape.rank.get_length()} rank"
                    f" is not equal to provided example_input rank {data_rank}")

            input_shape = user_input_shape if user_input_shape is not None else PartialShape([-1] * data_rank)
            update_list_or_dict(data_types, input_name, input_id,
                                example_dtype if example_dtype is not None else None)
            update_list_or_dict(input_shapes, input_name, input_id, input_shape)
    else:
        for input_id, example_input in enumerate(list_inputs):
            dtype = getattr(example_input, "dtype", type(example_input))
            ov_dtype = pt_to_ov_type_map.get(str(dtype))
            data_rank = getattr(example_input, "ndim", 0)
            input_shape = PartialShape([-1] * data_rank)
            input_name = input_names[input_id] if input_names else None
            update_list_or_dict(input_shapes, input_name, input_id, input_shape)
            update_list_or_dict(data_types, input_name, input_id, ov_dtype if ov_dtype is not None else None)

    args.placeholder_data_types = data_types
    args.placeholder_shapes = input_shapes
    if not args.input and input_names:
        args.input_list = input_names
        args.input = ",".join(input_names)


# pylint: disable=no-member
def to_torch_tensor(tensor):
    import torch  # pylint: disable=import-error
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    if isinstance(tensor, Tensor):
        return torch.tensor(tensor.data)
    if isinstance(tensor, (float, int, bool)):
        return tensor
    if isinstance(tensor, (tuple, list)):
        # TODO: Function to_torch_tensor should be renamed as it handles not only a tensor
        return tuple(to_torch_tensor(x) for x in tensor)
    if isinstance(tensor, dict) and all(isinstance(k, str) for k in tensor.keys()):
        return dict((k, to_torch_tensor(x)) for k, x in tensor.items())
    else:
        raise Error("Unexpected type of example_input. Supported types torch.Tensor, np.array or ov.Tensor. "
                    "Got {}".format(type(tensor)))


def prepare_torch_inputs(example_inputs):
    inputs = None
    if example_inputs is not None:
        inputs = example_inputs
        if isinstance(inputs, list):
            inputs = [to_torch_tensor(x) for x in inputs]
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
    else:
        # No example_input were provided, decoder will use scripting
        return None
    return inputs
