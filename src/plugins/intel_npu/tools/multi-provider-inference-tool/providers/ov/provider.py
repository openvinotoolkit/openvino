#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from collections import defaultdict
from collections.abc import Mapping
import math
import numpy as np
import os

import cv2

from providers.interfaces import Context
from providers.interfaces import Provider
from providers.interfaces import ProviderHolder
from params import Config
from params import ModelInfo
from params import TensorInfo
import utils

import openvino as ov

#-S-
import providers.ov.test

def ov_layout_to_string(layout : ov.Layout):
    return ''.join(layout.to_string()[1:-1].split(","))

class OVContextBase:
    def __init__(self):
        super().__init__()
        self.core = ov.Core()

    def read_model(self, model_path: str):
        model = self.core.read_model(model_path)
        return model

    def preprocess_model(self, model: ov.Model, preprocess_model_data: ModelInfo):
        if preprocess_model_data is None:
            return model

        # resired_shape_inputs = defaultdict(ov.PartialShape)
        desired_shape_inputs = {
            input_name: ov.PartialShape(desired_input_data["shape"])
            for input_name, desired_input_data in preprocess_model_data.preproc_per_io.items()
            if "shape" in desired_input_data.keys()
        }
        model.reshape(desired_shape_inputs)

        ppp = ov.preprocess.PrePostProcessor(model)
        for (
            input_name,
            desired_input_data,
        ) in preprocess_model_data.preproc_per_io.items():
            if "layout" in desired_input_data.keys():
                ppp.input(input_name).model().set_layout(
                    ov.Layout(desired_input_data["layout"])
                )
            if "element_type" in desired_input_data.keys():
                dtype = np.dtype(getattr(np, desired_input_data["element_type"]))
                ppp.input(input_name).model().set_element_type(ov.Type(dtype))
        model = ppp.build()
        return model

    @staticmethod
    def extract_shape_from_model_input(model_input):
        # OV can'nt obtain a shape from a node using its API
        # providing the node has a dynamic shape.
        # In this case we get shape through node attributes
        if 'shape' in model_input.get_node().get_attributes().keys():
            return model_input.get_node().get_attributes()["shape"]
        try:
            return model_input.get_shape()
        except Exception as ex:
            pass
        return ['...']

    @staticmethod
    def get_model_info(model) -> ModelInfo:
        info = ModelInfo()
        for minput in model.inputs:
            info.insert_info(
                minput.get_any_name(),
                {
                    "element_type": minput.get_element_type().to_dtype().name,
                    "shape": list(OVContextBase.extract_shape_from_model_input(minput)),
                    "node_type" : "input",
                },
            )
        for moutput in model.outputs:
            info.insert_info(
                moutput.get_any_name(),
                {
                    "element_type": moutput.get_element_type().to_dtype().name,
                    "shape": list(OVContextBase.extract_shape_from_model_input(moutput)),
                    "node_type" : "output",
                },
            )
        return info

    @staticmethod
    def get_common_tensor_info(tensor):
        info = TensorInfo()
        info.info["bytes_size"] = tensor.byte_size#bytearray(tensor.data)
        info.info["data"] = bytearray(tensor.data)
        info.info["element_type"] = tensor.get_element_type().to_dtype().name
        info.info["shape"] = list(tensor.get_shape())
        return info

    @staticmethod
    def collect_layouts_per_io(
        model: ov.Model, preprocess_model_data: ModelInfo
    ) -> dict:
        return_layouts = {}
        for model_input in model.inputs:
            model_input_name = model_input.get_any_name()
            if model_input_name in preprocess_model_data.preproc_per_io.keys() and "layout" in preprocess_model_data.preproc_per_io[model_input_name].keys():
                return_layouts[model_input_name] = ov_layout_to_string(ov.Layout(
                    preprocess_model_data.preproc_per_io[model_input_name]["layout"]
                ))
            else:
                shape = OVContextBase.extract_shape_from_model_input(model_input)
                return_layouts[model_input_name] = ov_layout_to_string(getLayoutByShape(shape))

        return return_layouts


def getLayoutByShape(shape: ov.Shape) -> ov.Layout:
    layout_str = utils.getLayoutByShape(list(shape))
    return ov.Layout.scalar() if len(layout_str) == 0 else ov.Layout(layout_str)


class OVImplProvider:
    def __init__(self, ctx: OVContextBase, model_path: str):
        self.model_path = model_path
        self.model = ctx.read_model(model_path)
        self.ctx = ctx
        self.layout_per_input = {}

    def init_model(self, preprocess_model_data: ModelInfo) -> Provider:
        self.model = self.ctx.preprocess_model(self.model, preprocess_model_data)

        # For some reasons OV doesn't remember a layout for each input in their ov.Model
        # We need this information of a preparation input tensors phase.
        # So that either we extract this layout from `preprocess_model_data` or
        # try to guess its format from model shape (legacy)
        self.layout_per_input = self.ctx.collect_layouts_per_io(
            self.model, preprocess_model_data
        )

    def get_model_info(self):
        info = self.ctx.get_model_info(self.model)
        info.set_model_name(utils.get_model_name(self.model_path))
        # for some unknown reasons ov.Model doesn't contain layout, so add it
        for i,l in self.layout_per_input.items():
            info.update_info(i, {"layout" : l})
        return info

    def get_tensor_info(self, tensor) -> TensorInfo:
        info = self.ctx.get_common_tensor_info(tensor)
        info.info["model"] = utils.get_model_name(self.model_path)
        info.validate()
        return info

    def prepare_input_tensors(self, input_files):
        return_tensors = {}
        model_input_files_input_pairs = []
        model_info = self.ctx.get_model_info(self.model)
        for model_input in self.model.inputs:
            model_input_name = model_input.get_any_name()
            if model_input_name not in input_files.keys():
                raise RuntimeError(
                    f"Cannot find input file for model input: {model_input_name} in user specified inputs: {input_files.keys()}"
                )

            infiles_description = utils.prepare_input_description(input_files[model_input_name], list(model_info.get_model_io_info(model_input_name)["shape"]), model_info.get_model_io_info(model_input_name)["element_type"], self.layout_per_input[model_input_name])
            tensor_raw_array, infiles_description = utils.load_objects_from_file(infiles_description)
            return_tensors[model_input_name] = ov.Tensor(tensor_raw_array)# -S- , ov.Shape(shape))
        return return_tensors

    def infer(self, executional_model, input_tensors):
        tensors_list = []
        for model_input in self.model.inputs:
            tensors_list.append(input_tensors[model_input.get_any_name()])

        results = executional_model(tensors_list)

        output_tensors = {}
        output_index = 0
        for output in self.model.outputs:
            output_tensors[output.get_any_name()] = ov.Tensor(results[output_index])
            output_index +=1
        return output_tensors


class OVCPUProvider(Provider):
    def __init__(self, ctx: OVContextBase, endpoint_full_name: str, model_path: str):
        super().__init__()
        self.impl = OVImplProvider(ctx, model_path)
        self.comp = None
        self.endpoint_full_name = endpoint_full_name

    @staticmethod
    def name() -> str:
        return "CPU$"

    def create_model(self, preprocess_model_data: ModelInfo, provider_config : Config) -> Provider:
        self.impl.init_model(preprocess_model_data)

        if provider_config.cfg_dict and len(provider_config.cfg_dict) != 0:
            self.comp = self.impl.ctx.core.compile_model(self.impl.model, self.endpoint_full_name, provider_config.cfg_dict)
        else:
            self.comp = self.impl.ctx.core.compile_model(self.impl.model, self.endpoint_full_name)
        return self

    def get_model_info(self):
        return self.impl.get_model_info()

    def get_tensor_info(self, tensor) -> TensorInfo:
        return self.impl.get_tensor_info(tensor)

    def prepare_input_tensors(self, input_files):
        return self.impl.prepare_input_tensors(input_files)

    def infer(self, input_tensors):
        return self.impl.infer(self.comp, input_tensors)


class OVGPUProvider(Provider):
    def __init__(self, ctx: OVContextBase, endpoint_full_name: str, model_path: str):
        super().__init__()
        self.impl = OVImplProvider(ctx, model_path)
        self.comp = None
        self.endpoint_full_name = endpoint_full_name

    @staticmethod
    def name() -> str:
        return "GPU((\\.(.+)$)|$)"

    def create_model(self, preprocess_model_data: ModelInfo, provider_config : Config) -> Provider:
        self.impl.init_model(preprocess_model_data)

        if provider_config.cfg_dict and len(provider_config.cfg_dict) != 0:
            self.comp = self.impl.ctx.core.compile_model(self.impl.model, self.endpoint_full_name, provider_config.cfg_dict)
        else:
            self.comp = self.impl.ctx.core.compile_model(self.impl.model, self.endpoint_full_name)
        return self

    def get_model_info(self):
        return self.impl.get_model_info()

    def get_tensor_info(self, tensor) -> TensorInfo:
        return self.impl.get_tensor_info(tensor)

    def prepare_input_tensors(self, input_files):
        return self.impl.prepare_input_tensors(input_files)

    def infer(self, input_tensors):
        return self.impl.infer(self.comp, input_tensors)


class OVNPUProvider(Provider):
    def __init__(self, ctx: OVContextBase, endpoint_full_name: str, model_path: str):
        super().__init__()
        self.impl = OVImplProvider(ctx, model_path)
        self.comp = None
        self.endpoint_full_name = endpoint_full_name

    @staticmethod
    def name() -> str:
        return "NPU((\\.(.+)$)|$)"

    def create_model(self, preprocess_model_data: ModelInfo, provider_config : Config) -> Provider:
        self.impl.init_model(preprocess_model_data)

        if provider_config.cfg_dict and len(provider_config.cfg_dict) != 0:
            self.comp = self.impl.ctx.core.compile_model(self.impl.model, self.endpoint_full_name, provider_config.cfg_dict)
        else:
            self.comp = self.impl.ctx.core.compile_model(self.impl.model, self.endpoint_full_name)
        return self

    def get_model_info(self):
        return self.impl.get_model_info()

    def get_tensor_info(self, tensor) -> TensorInfo:
        return self.impl.get_tensor_info(tensor)

    def prepare_input_tensors(self, input_files):
        return self.impl.prepare_input_tensors(input_files)

    def infer(self, input_tensors):
        return self.impl.infer(self.comp, input_tensors)

class OVContext(Context):
    ov_registered_providers = ProviderHolder([OVCPUProvider, OVGPUProvider, OVNPUProvider])

    def __init__(self, provider_name):
        super().__init__()
        self.ov_ctx = OVContextBase()

        provider_name_specific, prefix = ProviderHolder.__remove_provider_prefix__(
            provider_name
        )
        if prefix != "ov":
            raise RuntimeError(
                f'Incorrect prefix: {prefix} of the provider name: {provider_name} - "ov" expected.'
            )

        self.provider_fabric, device = OVContext.ov_registered_providers.get_provider_by_name(
            provider_name_specific
        )
        self.creator = lambda model_name: self.provider_fabric(self.ov_ctx, device, model_name)

    def create_provider(self, model_path: str):
        provider = self.creator(model_path)
        return provider

    @staticmethod
    def provider_names() -> list:
        return OVContext.ov_registered_providers.prefixed_names("ov")


Provider.register(OVCPUProvider)
Provider.register(OVGPUProvider)
Provider.register(OVNPUProvider)
Context.register(OVContext)


def provider_names() -> list:
    return OVContext.provider_names()

def create(provider_name):
    return OVContext(provider_name)
