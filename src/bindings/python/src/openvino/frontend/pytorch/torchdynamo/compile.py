# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import os
import torch
import torch.overrides

from torch.fx import GraphModule

from openvino.frontend import FrontEndManager
from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
from openvino.runtime import Core, Type, PartialShape, serialize

from typing import Callable, Optional


def openvino_compile(gm: GraphModule, *args, model_hash_str: str = None):
    core = Core()

    device = "CPU"

    if os.getenv("OPENVINO_TORCH_BACKEND_DEVICE") is not None:
        device = os.getenv("OPENVINO_TORCH_BACKEND_DEVICE")
        assert device in core.available_devices, "Specified device " + device + " is not in the list of OpenVINO Available Devices"

    file_name = None
    cache_root = "./cache/"
    if os.getenv("OPENVINO_TORCH_CACHE_DIR") is not None:
        cache_root = os.getenv("OPENVINO_TORCH_CACHE_DIR")
    if model_hash_str is not None:
        model_cache_dir = cache_root + "/model/"
        try:
            os.makedirs(model_cache_dir, exist_ok=True)
            file_name = model_cache_dir + model_hash_str + "_" + device
        except OSError as error:
            print("Cache directory ", cache_root, " cannot be created. Model caching is disabled. Error: ", error)
            file_name = None
            model_hash_str = None

    if file_name is not None and os.path.isfile(file_name + ".xml") and os.path.isfile(file_name + ".bin"):
        om = core.read_model(file_name + ".xml")
    else:
        fe_manager = FrontEndManager()
        fe = fe_manager.load_by_framework("pytorch")

        input_shapes = []
        input_types = []
        for idx, input_data in enumerate(args):  # subgraph.example_inputs):
            input_types.append(input_data.type())
            input_shapes.append(input_data.size())

        decoder = TorchFXPythonDecoder(gm, gm, input_shapes=input_shapes, input_types=input_types)

        im = fe.load(decoder)

        om = fe.convert(im)

        if file_name is not None:
            serialize(om, file_name + ".xml", file_name + ".bin")

    dtype_mapping = {
        torch.float32: Type.f32,
        torch.float64: Type.f64,
        torch.float16: Type.f16,
        torch.int64: Type.i64,
        torch.int32: Type.i32,
        torch.uint8: Type.u8,
        torch.int8: Type.i8,
        torch.bool: Type.boolean
    }

    for idx, input_data in enumerate(args):
        om.inputs[idx].get_node().set_element_type(dtype_mapping[input_data.dtype])
        om.inputs[idx].get_node().set_partial_shape(PartialShape(list(input_data.shape)))
    om.validate_nodes_and_infer_types()

    if model_hash_str is not None:
        core.set_property({'CACHE_DIR': cache_root + '/blob'})

    compiled = core.compile_model(om, device)
    return compiled
