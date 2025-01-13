# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference engine runners."""
# pylint:disable=import-error
import logging as log
import os
import platform
import sys
from pprint import pformat

import numpy as np
from e2e_tests.utils.test_utils import align_input_names, get_shapes_with_frame_size
from e2e_tests.utils.test_utils import get_infer_result

try:
    import resource

    mem_info_available = True
except ImportError:
    mem_info_available = False

from openvino.runtime import Core
from openvino.inference_engine import get_version as ie_get_version
from e2e_tests.common.multiprocessing_utils import multiprocessing_run

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

from e2e_tests.common.infer.provider import ClassProvider
from e2e_tests.common.infer.network_modifiers import Container


def resolve_library_name(libname):
    """Return platform-specific library name given basic libname."""
    if not libname:
        return libname
    if os.name == 'nt':
        return libname + '.dll'
    if platform.system() == 'Darwin':
        return 'lib' + libname + '.dylib'
    return 'lib' + libname + '.so'


def parse_device_name(device_name):
    device_name_ = device_name
    if "HETERO:" in device_name:
        device_name_ = "HETERO"
    elif "MULTI:" in device_name:
        device_name_ = "MULTI"
    elif ("AUTO:" in device_name) or ("AUTO" == device_name):
        device_name_ = "AUTO"
    elif "BATCH:" in device_name:
        device_name_ = "BATCH"
    else:
        device_name_ = device_name

    return device_name_


class Infer(ClassProvider):
    """Basic inference engine runner."""
    __action_name__ = "ie_sync"

    def __init__(self, config):
        self.device = parse_device_name(config["device"])
        self.timeout = config.get("timeout", 300)
        self.res = None
        self.network_modifiers = Container(config=config.get("network_modifiers", {}))
        self.plugin_cfg = config.get("plugin_config", {})
        self.plugin_cfg_target_device = config.get("plugin_cfg_target_device", self.device)
        self.consecutive_infer = config.get("consecutive_infer", False)
        self.index_infer = config.get('index_infer')
        self.xml = None
        self.bin = None
        self.model_path = None

    def _get_thermal_metric(self, exec_net, ie):
        if "MYRIAD" in self.device:
            supported_metrics = exec_net.get_property("SUPPORTED_METRICS")
            if "DEVICE_THERMAL" in supported_metrics:
                return round(exec_net.get_property("DEVICE_THERMAL"), 3)
            else:
                log.warning("Expected metric 'DEVICE_THERMAL' doesn't present in "
                            "supported metrics list {} for MYRIAD plugin".format(supported_metrics))
                return None
        elif "HDDL" in self.device:
            # TODO: Uncomment when HDDL plugin will support 'SUPPORTED_METRICS' metric and remove try/except block
            # supported_metrics = ie.get_metric("HDDL", "SUPPORTED_METRICS")
            # if "DEVICE_THERMAL" in supported_metrics:
            #     return ie.get_metric("HDDL", "VPU_HDDL_DEVICE_THERMAL")
            # else:
            #     log.warning("Expected metric 'DEVICE_THERMAL' doesn't present in "
            #                 "supported metrics list {} for HDDL plugin".format(supported_metrics))
            #     return None
            try:
                return [round(t, 3) for t in ie.get_property("HDDL", "VPU_HDDL_DEVICE_THERMAL")]
            except RuntimeError:
                log.warning("Failed to query metric 'VPU_HDDL_DEVICE_THERMAL' for HDDL plugin")
                return None

        else:
            return None

    def _configure_plugin(self, ie):
        if self.plugin_cfg:
            supported_props = ie.get_property(self.plugin_cfg_target_device, 'SUPPORTED_PROPERTIES')
            if 'INFERENCE_PRECISION_HINT' not in supported_props:
                log.warning(
                    f'inference precision hint is not supported for device {self.plugin_cfg_target_device},'
                    f' option will be ignored')
                return
            log.info("Setting config to the {} plugin. \nConfig:\n{}".format(self.plugin_cfg_target_device,
                                                                             pformat(self.plugin_cfg)))
            ie.set_property(self.plugin_cfg_target_device, self.plugin_cfg)

    def _infer(self, input_data):
        log.info("Inference Engine version: {}".format(ie_get_version()))
        log.info("Using API v2.0")
        result, load_net_to_plug_time = None, None
        if mem_info_available:
            mem_usage_in_kbytes_before_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        log.info("Creating Core Engine...")
        ie = Core()
        self._configure_plugin(ie)

        log.info("Loading network files")

        if self.model_path:
            self.ov_model = ie.read_model(model=self.model_path)
        if self.xml:
            self.ov_model = ie.read_model(model=self.xml)
        self.network_modifiers.execute(network=self.ov_model, input_data=input_data)

        log.info("Loading network to the {} device...".format(self.device))
        compiled_model = ie.compile_model(self.ov_model, self.device)

        for input_tensor in self.ov_model.inputs:
            # all input and output tensors have to be named
            assert input_tensor.names, "Input tensor {} has no names".format(input_tensor)

        result = []
        if self.consecutive_infer:
            for infer_run_counter in range(2):
                helper = get_infer_result(input_data[infer_run_counter], compiled_model, self.ov_model,
                                          infer_run_counter, self.index_infer)
                result.append(helper)
        else:
            infer_result = get_infer_result(input_data, compiled_model, self.ov_model, index_infer=self.index_infer)
            result.append(infer_result)

        if not self.consecutive_infer:
            result = result[0]

        if mem_info_available:
            mem_usage_in_kbytes_after_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem_usage_ie = round((mem_usage_in_kbytes_after_run - mem_usage_in_kbytes_before_run) / 1024)
        else:
            mem_usage_ie = -1

        if "exec_net" in locals():
            del compiled_model
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, mem_usage_ie

    def infer(self, input_data):
        self.res, self.load_net_to_plug_time, self.mem_usage_ie = \
            multiprocessing_run(self._infer, [input_data], "Inference Engine Python API", self.timeout)

        return self.res


class SequenceInference(Infer):
    """Sequence inference engine runner."""
    __action_name__ = "ie_sequence"

    def __init__(self, config):
        super().__init__(config=config)
        self.default_shapes = config.get('default_shapes')

    def _infer(self, input_data):
        log.info("Inference Engine version: {}".format(ie_get_version()))
        log.info("Using API v2.0")
        result, load_net_to_plug_time = None, None
        if mem_info_available:
            mem_usage_in_kbytes_before_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        log.info("Creating Core Engine...")
        ie = Core()
        self._configure_plugin(ie)

        log.info("Loading network files")
        if self.model_path:
            ov_model = ie.read_model(model=self.model_path)
        else:
            ov_model = ie.read_model(model=self.xml)
        self.network_modifiers.execute(network=ov_model, input_data=input_data)

        log.info("Loading network to the {} device...".format(self.device))
        compiled_model = ie.compile_model(ov_model, self.device)

        for input_tensor in ov_model.inputs:
            # all input and output tensors have to be named
            assert input_tensor.names, "Input tensor {} has no names".format(input_tensor)

        result = []
        input_data = align_input_names(input_data, ov_model)
        # make input_data (dict) a list of frame feed dicts
        input_data = get_shapes_with_frame_size(self.default_shapes, ov_model, input_data)

        new_input = []
        num_frames = max([input_data[key].shape[0] for key in input_data])
        input_data = {key: value if value.shape[0] == num_frames else np.tile(value, num_frames).reshape(num_frames, *(
            list(value.shape)[1:])) for key, value in input_data.items()}
        log.info("Total number of input frames: {}".format(num_frames))

        for current_frame_index in range(0, num_frames):
            cur_frame_data = {key: value[current_frame_index] for key, value in input_data.items()}
            infer_result = get_infer_result(cur_frame_data, compiled_model, ov_model, current_frame_index)
            result.append(infer_result)

        # make result (list of infer result for each frame) a dict (each layer contains infer result for all frames)
        result = {key: [value[key] for value in result] for key in result[0]}
        result = {key: np.stack(values, axis=0).reshape(num_frames, *(list(values[0].shape[1:]))) for key, values in
                  result.items()}

        if mem_info_available:
            mem_usage_in_kbytes_after_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem_usage_ie = round((mem_usage_in_kbytes_after_run - mem_usage_in_kbytes_before_run) / 1024)
        else:
            mem_usage_ie = -1

        if "exec_net" in locals():
            del compiled_model
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, mem_usage_ie

    def infer(self, model):
        self.res, self.load_net_to_plug_time, self.mem_usage_ie = \
            multiprocessing_run(self._infer, [model], "Inference Engine Python API", self.timeout)

        return self.res
