#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import copy
import json
import onnx
import onnxruntime
import os
import pathlib
import re
import sys
import numpy as np

from providers.interfaces import Context
from providers.interfaces import Provider
from providers.interfaces import ProviderHolder
from common.provider_description import Config, ModelInfo, TensorInfo

import utils

def onnx_type_to_dtype(onnx_type):
    conv_map = { "float" : "float32",
                "double" : "float64"}

    ttype =  onnx_type.replace("tensor","").replace("(","").replace(")","")
    if ttype in conv_map.keys():
        ttype = conv_map[ttype]

    return ttype #helper.tensor_dtype_to_np_dtype(onnx_type)

class OnnxContextBase:
    def __init__(self):
        pass

    @staticmethod
    def get_model_info(session) -> ModelInfo:
        info = ModelInfo()
        for minput in session.get_inputs():
            info.insert_info(
                minput.name,
                {
                    "element_type": onnx_type_to_dtype(minput.type),
                    "shape": minput.shape,
                    "node_type" : "input",
                },
            )
        for moutput in session.get_outputs():
            info.insert_info(
                moutput.name,
                {
                    "element_type": onnx_type_to_dtype(moutput.type),
                    "shape": moutput.shape,
                    "node_type" : "output",
                },
            )
        return info

    @staticmethod
    def get_common_tensor_info(tensor):
        info = TensorInfo()
        info.info["bytes_size"] = 0#tensor.byte_size#bytearray(tensor.data)
        info.info["data"] = bytearray(tensor.numpy())
        info.info["element_type"] = onnx_type_to_dtype(tensor.data_type())
        info.info["shape"] = tensor.shape()
        return info

    @staticmethod
    def collect_layouts_per_io(session, preprocess_model_data: ModelInfo) -> dict:
        return_layouts = {}
        for model_input in session.get_inputs():
            model_input_name = model_input.name
            if model_input_name in preprocess_model_data.preproc_per_io.keys() and "layout" in preprocess_model_data.preproc_per_io[model_input_name].keys():
                return_layouts[model_input_name] = preprocess_model_data.preproc_per_io[model_input_name]["layout"]
            else:
                return_layouts[model_input_name] = utils.getLayoutByShape(model_input.shape)
        return return_layouts

    @staticmethod
    def get_session_options(options, onnxruntime_config_option_dict):
        options.log_severity_level = 3
        for attr in onnxruntime_config_option_dict.keys():
            match attr:
                case "log_severity_level":
                    options.log_severity_level = int(onnxruntime_config_option_dict[attr])
                case "log_verbosity_level":
                    options.log_verbosity_level = int(onnxruntime_config_option_dict[attr])
                case "graph_optimization_level":
                    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(int(onnxruntime_config_option_dict[attr]))
                case "session.disable_cpu_ep_fallback":
                    options.add_session_config_entry("session.disable_cpu_ep_fallback", str(onnxruntime_config_option_dict[attr]))
        return options

    @staticmethod
    def get_provider_config(provider_name, onnxruntime_config_option_dict):
        for candidate_provider in onnxruntime_config_option_dict.keys():
            if re.search(provider_name, candidate_provider):
                return copy.deepcopy(onnxruntime_config_option_dict[candidate_provider])

        # TODO backward compatibility with scale-tools preserving
        # return the entire config in case of provider didn't set
        # as the current version doesn't support per-provider options
        return copy.deepcopy(onnxruntime_config_option_dict)

def preprocess_model_static_shapes(model, preprocess_model_data: ModelInfo):
    if preprocess_model_data is None:
        return
    for model_input in model.graph.input:
        model_input_name = model_input.name
        if model_input_name in preprocess_model_data.preproc_per_io.keys():
            if "shape" in preprocess_model_data.preproc_per_io[model_input_name].keys():
                tensor_type = model_input.type.tensor_type
                del tensor_type.shape.dim[:]
                for dim in preprocess_model_data.preproc_per_io[model_input_name]["shape"]:
                    dim_proto = tensor_type.shape.dim.add()
                    dim_proto.dim_value = dim
            if "element_type" in preprocess_model_data.preproc_per_io[model_input_name].keys():
                tensor_type = model_input.type.tensor_type
                tensor_type.elem_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(getattr(np, preprocess_model_data.preproc_per_io[model_input_name]["element_type"])))
                # TODO think about how we can change element_type
                # raise RuntimeError("setting 'element_type' is unsupported for ONNX models (probably we should traverse entire gpraph and overwrite 'elem_type' in each node)")
                #print(f"setting 'element_type': {tensor_type.elem_type} is unsupported for ONNX models (probably we should traverse entire gpraph and overwrite 'elem_type' in each node)", file=sys.stderr)
                print(f"setting 'element_type': {tensor_type.elem_type} is unsupported for ONNX models (probably we should traverse entire gpraph and overwrite 'elem_type' in each node)")
    return model

class CPUExecutionProvider(Provider):
    def __init__(self, ctx : OnnxContextBase, endpoint_full_name : str, model_path: str):
        super().__init__()
        self.ctx = ctx
        self.model_path = model_path
        self.session = None
        self.endpoint_full_name = endpoint_full_name

    @staticmethod
    def name() -> str:
        return "CPUExecutionProvider$"

    def create_model(self, preprocessing_request_data, provider_config : Config) -> Provider:
        model = onnx.load(self.model_path)
        model = preprocess_model_static_shapes(model, preprocessing_request_data)

        options = onnxruntime.SessionOptions()
        options = OnnxContextBase.get_session_options(options, provider_config.cfg_dict)
        external_provider_dict_options = OnnxContextBase.get_provider_config(
                                            CPUExecutionProvider.name(),
                                            provider_config.cfg_dict
                                        )
        self.session = onnxruntime.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"],
            provider_options=[external_provider_dict_options]
        )

        self.layout_per_input = OnnxContextBase.collect_layouts_per_io(self.session, preprocessing_request_data)
        return self

    def get_model_info(self) -> ModelInfo:
        info = OnnxContextBase.get_model_info(self.session)
        info.set_model_name(utils.get_model_name(self.model_path))
        # for some unknown reasons onnx session doesn't contain layout, so add it
        for i,l in self.layout_per_input.items():
            info.update_info(i, {"layout" : l})
        return info

    def get_tensor_info(self, tensor) -> TensorInfo:
        info = OnnxContextBase.get_common_tensor_info(tensor)
        info.info["model"] = utils.get_model_name(self.model_path)
        info.validate()
        return info

    def prepare_input_tensors(self, input_files):
        return_tensors = {}
        model_input_files_input_pairs = []
        model_info = OnnxContextBase.get_model_info(self.session)
        for model_input in self.session.get_inputs():
            model_input_name = model_input.name
            if model_input_name not in input_files.keys():
                raise RuntimeError(
                    f"Cannot find input file for model input: {model_input_name} in user specified inputs: {input_files.keys()}"
                )

            infiles_description = utils.prepare_input_description(input_files[model_input_name], list(model_info.get_model_io_info(model_input_name)["shape"]), model_info.get_model_io_info(model_input_name)["element_type"], self.layout_per_input[model_input_name])
            tensor_raw_array, infiles_description = utils.load_objects_from_file(infiles_description)
            return_tensors[model_input_name] = onnxruntime.OrtValue.ortvalue_from_numpy(tensor_raw_array)
        return return_tensors

    def infer(self, tensors_dict):
        output_names = [output.name for output in self.session.get_outputs()]
        results = self.session.run(output_names, tensors_dict)
        return {name:onnxruntime.OrtValue.ortvalue_from_numpy(t) for (name, t) in zip(output_names,results)}

def is_ovep_shape_static(ovep_shape):
    return  all(isinstance(d, int) for d in ovep_shape)

def preprocess_model_ovep_shapes(model, preprocess_model_data: ModelInfo):
    if preprocess_model_data is None:
        return

    # check if preprocessor requests static shapes only before changing a model
    is_dynamic_shape_requested = False
    expected_dynamic_shapes= {}
    for model_input_name in preprocess_model_data.preproc_per_io.keys():
        if "shape" in preprocess_model_data.preproc_per_io[model_input_name].keys():
            if not is_ovep_shape_static(preprocess_model_data.preproc_per_io[model_input_name]["shape"]):
                is_dynamic_shape_requested = True

            # remember shape regardless it's static or dynamic
            expected_dynamic_shapes[model_input_name] = preprocess_model_data.preproc_per_io[model_input_name]["shape"]

    for model_input in model.graph.input:
        model_input_name = model_input.name
        if model_input_name in preprocess_model_data.preproc_per_io.keys():
            if "shape" in preprocess_model_data.preproc_per_io[model_input_name].keys():
                if is_dynamic_shape_requested:
                    # do not change shape if at least one dynamic shape has been requested
                    # ONNX receives only Int dimension values, so we cannot set either '?' or [min,max]
                    # as it is supposed to be OVEP,
                    # we will use 'reshape_input' later as a part of provider configs.
                    # In this case we collect ALL requested shapes as dictionary,
                    # so that we will construct a proper provider config later
                    continue

                tensor_type = model_input.type.tensor_type
                del tensor_type.shape.dim[:]
                for dim in preprocess_model_data.preproc_per_io[model_input_name]["shape"]:
                    dim_proto = tensor_type.shape.dim.add()
                    dim_proto.dim_value = dim
            if "element_type" in preprocess_model_data.preproc_per_io[model_input_name].keys():
                tensor_type = model_input.type.tensor_type
                tensor_type.elem_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(getattr(np, preprocess_model_data.preproc_per_io[model_input_name]["element_type"])))

                # TODO think about how we can change element_type
                # raise RuntimeError("setting 'element_type' is unsupported for ONNX models (probably we should traverse entire gpraph and overwrite 'elem_type' in each node)")
                print("setting 'element_type' is unsupported for ONNX models (probably we should traverse entire gpraph and overwrite 'elem_type' in each node)")

    return model, expected_dynamic_shapes if is_dynamic_shape_requested else {}

def apply_reshape_input_config_param(provider_cfg, expected_dynamic_shapes):
    if len(expected_dynamic_shapes) == 0:
        return provider_cfg
    if "reshape_input" in provider_cfg.keys():
        raise RuntimeError(f"Provider config already has the item \"reshape_input\" and its the value is: {provider_cfg['reshape_input']}. As a model preprocessing config is specified and the model inputs have \"shape\" parameters: {expected_dynamic_shapes}, this is considered as ambiguity. Please resolve the discrepancy either by removing \"shape\" from the model preprocessing config or exclude \"reshape_input\" from the provider configuration")
    provider_cfg["reshape_input"] = ",".join([str(i) + str(s) for i,s in expected_dynamic_shapes.items()])
    return provider_cfg

class OVEPCPU(CPUExecutionProvider):
    def __init__(self, ctx : OnnxContextBase, endpoint_full_name : str, model_path: str):
        super().__init__(ctx, endpoint_full_name, model_path)
        self.endpoint_full_name = endpoint_full_name

    @staticmethod
    def name() -> str:
        return "CPU$"

    def create_model(self, preprocessing_request_data, provider_config : Config) -> Provider:
        model = onnx.load(self.model_path)
        model, expected_dynamic_shapes = preprocess_model_ovep_shapes(model, preprocessing_request_data)

        options = onnxruntime.SessionOptions()
        options.add_session_config_entry(
            "session.disable_cpu_ep_fallback",
            "1"
        )
        # Extract generic onnx options from config
        options = OnnxContextBase.get_session_options(options, provider_config.cfg_dict)

        # add necessary provider options so that OVEP
        # can use it in a safe and non-controversial manner
        external_provider_dict_options = OnnxContextBase.get_provider_config(
                                            OVEPCPU.name(),
                                            provider_config.cfg_dict
                                        )
        if "device_type" in external_provider_dict_options.keys():
            raise RuntimeError(f"ERROR: Provider options conflict: \"device_type\" must not be specified in options: {external_provider_dict_options}")
        external_provider_dict_options["device_type"] = self.endpoint_full_name

        # specify model dynamism if necessary
        external_provider_dict_options = apply_reshape_input_config_param(external_provider_dict_options, expected_dynamic_shapes)
        self.session = onnxruntime.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["OpenVINOExecutionProvider"],
            provider_options=[external_provider_dict_options]
        )

        self.layout_per_input = OnnxContextBase.collect_layouts_per_io(self.session, preprocessing_request_data)
        return self

class OVEPNPU(CPUExecutionProvider):
    def __init__(self, ctx : OnnxContextBase, endpoint_full_name: str, model_path: str):
        super().__init__(ctx, endpoint_full_name, model_path)
        self.endpoint_full_name = endpoint_full_name

    @staticmethod
    def name() -> str:
        return "NPU((\\.(.+)$)|$)"

    def create_model(self, preprocessing_request_data, provider_config : Config) -> Provider:
        model = onnx.load(self.model_path)
        model,expected_dynamic_shapes = preprocess_model_ovep_shapes(model, preprocessing_request_data)

        options = onnxruntime.SessionOptions()
        options.add_session_config_entry(
            "session.disable_cpu_ep_fallback",
            "1"
        )
        # Extract generic onnx options from config
        options = OnnxContextBase.get_session_options(options, provider_config.cfg_dict)

        # add necessary options so that OVEP can use it in a safe and non-controversial manner
        external_provider_dict_options = OnnxContextBase.get_provider_config(
                                            OVEPNPU.name(),
                                            provider_config.cfg_dict
                                        )
        if "device_type" in external_provider_dict_options.keys():
            raise RuntimeError(f"ERROR: Provider options conflict: \"device_type\" must not be specified in options: {external_provider_dict_options}")
        external_provider_dict_options["device_type"] = Provider.canonize_endpoint_name(self.endpoint_full_name)

        # OPENVINOExecution provider expects JSON object serialized in string
        # provided you specify 'load_config' value
        if "load_config" in external_provider_dict_options.keys():
            if os.path.isfile(external_provider_dict_options["load_config"]):
                with open(external_provider_dict_options["load_config"], "r") as file:
                    json_cfg = json.load(file)
                    external_provider_dict_options["load_config"] = json.dumps(json_cfg)

        # specify model dynamism if necessary
        external_provider_dict_options = apply_reshape_input_config_param(external_provider_dict_options, expected_dynamic_shapes)
        self.session = onnxruntime.InferenceSession(
            model.SerializeToString(), sess_options=options, providers=["OpenVINOExecutionProvider"],
            provider_options=[external_provider_dict_options]
        )

        self.layout_per_input = OnnxContextBase.collect_layouts_per_io(self.session, preprocessing_request_data)
        return self

class ONNXMediator(Context):
    registered_providers = ProviderHolder([OVEPCPU,OVEPNPU])

    def __init__(self):
        super().__init__()

    @staticmethod
    def provider_names() -> list:
        return ONNXMediator.registered_providers.prefixed_names("OpenVINOExecutionProvider")

    @staticmethod
    def get_provider_by_name(provider_name):
        return ONNXMediator.registered_providers.get_provider_by_name(provider_name)


class ONNXContext(Context):
    registered_providers = ProviderHolder([CPUExecutionProvider,ONNXMediator])

    def __init__(self, provider_name):
        super().__init__()
        self.ctx = OnnxContextBase()

        provider_name_specific, prefix = ProviderHolder.__remove_provider_prefix__(
            provider_name
        )
        if prefix != "onnx":
            raise RuntimeError(
                f'Incorrect prefix: {prefix} of the provider name: {provider_name} - "onnx" expected.'
            )

        self.provider_fabric, device = ONNXContext.registered_providers.get_provider_by_name(
            provider_name_specific
        )
        self.creator = lambda model_name: self.provider_fabric(self.ctx, device, model_name)

    def create_provider(self, model_path: str):
        provider = self.creator(model_path)
        return provider

    @staticmethod
    def provider_names() -> list:
        return ONNXContext.registered_providers.prefixed_names("onnx")


Provider.register(CPUExecutionProvider)
Provider.register(OVEPCPU)
Provider.register(OVEPNPU)
Context.register(ONNXMediator)
Context.register(ONNXContext)



def provider_names() -> list:
    return ONNXContext.provider_names()

def create(provider_name):
    return ONNXContext(provider_name)
