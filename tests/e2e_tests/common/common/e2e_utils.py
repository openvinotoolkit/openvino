# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.runtime import Core, Model
import torch
from typing import Any
import logging

log = logging.getLogger(__name__)


def collect_tensor_names(instance: Model, tensor_type_name: str, out: dict) -> dict:
    """
    @param instance: Read OpenVino model
    @param tensor_type_name: Type of tensors
    @param out: Dictionary for tensor names
    @return: Dictionary with collected tensor names
    """
    tensor_dicts = getattr(instance, tensor_type_name, None)
    assert tensor_dicts, f"Wrong tensor type name is used: {tensor_type_name}"
    for tensor in tensor_dicts:
        tensor_names = getattr(tensor, 'names', None)
        assert tensor_names, f"Tensor {tensor_type_name} must have 'names' field"
        for tensor_name in tensor_names:
            out[tensor_name] = tensor_name
    return out


def get_tensor_names_dict(xml_ir: Any) -> dict:
    """
    @param xml_ir: Path to xml part of IR
    @return: output dictionary with collected tensor names
    """
    log.debug(f"IR xml path: {xml_ir}")

    core = Core()
    ov_model = core.read_model(model=xml_ir)
    log.debug(f"Read OpenVino model: {ov_model}")

    out_dict = collect_tensor_names(ov_model, 'inputs', {})
    out_dict = collect_tensor_names(ov_model, 'outputs', out_dict)
    log.debug(f"Output dictionary with collected tensor names : {out_dict}")
    return out_dict


def mo_additional_args_static_dict(descriptor: dict, tensor_type) -> dict:
    """
    Convert input descriptor to MO additional static arguments dictionary like
    {"input": string with inputs name, "input_shape": string with inputs shape}
    @param descriptor: input descriptor as dict
    @param tensor_type: type of output tensors
    @return: MO additional arguments as dict
    """
    output_dict = {"example_input": []}
    for key in descriptor.keys():
        shape = descriptor[key].get('default_shape')
        output_dict["example_input"].append(torch.ones(shape, dtype=tensor_type))
    return output_dict


def mo_additional_args_static_str(input_descriptor: dict, port: Any = None, precision: int = 32) -> dict:
    """
    Convert input descriptor to MO additional static arguments with dict like
    {"input": inputs string with precision and shape}
    @param input_descriptor: input descriptor as dict
    @param precision: precision
    @param port: port if needed
    @return: MO additional arguments as dict
    """
    temp = ""
    precision = "{" + f"i{precision}" + "}"
    port = port if port else ""
    for k in input_descriptor.keys():
        temp += f"{k}{port}{precision}{str(input_descriptor[k]['default_shape']).replace(' ', '')},"
    return {"input": input[:-1]}

