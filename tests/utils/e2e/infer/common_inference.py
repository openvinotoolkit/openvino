"""Inference engine runners."""
# pylint:disable=import-error
import logging as log
import os
import platform
import sys
import time
from pprint import pformat

import numpy as np

from e2e_oss.utils.test_utils import align_input_names, get_shapes_with_frame_size
from e2e_oss.utils.test_utils import get_infer_result

try:
    import resource

    mem_info_available = True
except ImportError:
    mem_info_available = False

from openvino.runtime import Core
from openvino.inference_engine import IECore, get_version as ie_get_version
from e2e_oss.common_utils.multiprocessing_utils import multiprocessing_run
from e2e_oss.utils.path_utils import resolve_file_path

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)

from utils.e2e.infer.provider import ClassProvider
from utils.e2e.infer.network_modifiers import Container


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
        self.xml = None  # dynamically set xml file path
        self.bin = None  # dynamically set bin file path
        self.model_path = config.get("model_path")  # we infer original model without IR in ONNX Importer tests
        self.network_modifiers = Container(config=config.get("network_modifiers", {}))
        self.plugin_cfg = config.get("plugin_config", {})
        self.plugin_cfg_target_device = config.get("plugin_cfg_target_device", self.device)
        self.consecutive_infer = config.get("consecutive_infer", False)
        self.index_infer = config.get('index_infer')

    def _configure_plugin(self, ie):
        if self.plugin_cfg:
            log.info("Setting config to the {} plugin. \nConfig:\n{}".format(self.plugin_cfg_target_device,
                                                                             pformat(self.plugin_cfg)))
            ie.set_config(self.plugin_cfg, self.plugin_cfg_target_device)

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

    def _infer(self, input_data):
        log.info("Using old API")
        log.info("Inference Engine version: {}".format(ie_get_version()))
        result, load_net_to_plug_time = None, None
        if mem_info_available:
            mem_usage_in_kbytes_before_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        log.info("Creating IE Core Engine...")
        ie = IECore()
        self._configure_plugin(ie)

        log.info("Loading network files")
        if self.model_path:
            net = ie.read_network(model=self.model_path)
        else:
            net = ie.read_network(model=self.xml, weights=self.bin)
        self.network_modifiers.execute(network=net, input_data=input_data)

        log.info("Loading network to the {} device...".format(self.device))
        # Measure time of loading network to plugin
        t_load_to_pl = time.time()
        exec_net = ie.load_network(net, self.device)
        load_net_to_plug_time = time.time() - t_load_to_pl

        t0 = self._get_thermal_metric(exec_net=exec_net, ie=ie)
        if t0 is not None:
            log.info("{} device thermal metric before infer: \n{}".format(self.device, t0, 3))

        log.info("Starting inference")
        result = exec_net.infer(input_data)

        t1 = self._get_thermal_metric(exec_net=exec_net, ie=ie)
        if t1 is not None:
            if isinstance(t1, list):
                diff = [round(t - t0[i], 3) for i, t in enumerate(t1)]
            else:
                diff = round(t1 - t0, 3)
            log.info("{} device thermal metric after infer: {}. \t\nHeating: {}".format(self.device, t1, diff))

        if mem_info_available:
            mem_usage_in_kbytes_after_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem_usage_ie = round((mem_usage_in_kbytes_after_run - mem_usage_in_kbytes_before_run) / 1024)
        else:
            mem_usage_ie = -1

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, mem_usage_ie

    def infer(self, input_data):
        self.res, self.load_net_to_plug_time, self.mem_usage_ie = \
            multiprocessing_run(self._infer, [input_data], "Inference Engine Python API", self.timeout)

        return self.res


class InferAPI20(Infer):
    """Basic inference engine runner."""
    __action_name__ = "ie_sync_api_2"

    def __init__(self, config):
        super().__init__(config=config)

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


class AsyncInfer(Infer):
    """Basic inference engine runner."""
    __action_name__ = "ie_async"

    def __init__(self, config):
        super().__init__(config)
        self.nireq = config.get("num_requests", 2)
        self.multi_image = config.get("multi_image", False)

    def _infer(self, input_data):
        if self.multi_image:
            log.info("Multi-image mode enabled. Each request will be inferred with it's own input data.")
            assert isinstance(input_data, list) and all([isinstance(data, dict) for data in input_data]), \
                "For multi image scenario input_data has to be a list of dictionaries!"
            assert len(input_data) >= self.nireq, \
                "Len of input data has to be more or equal to number of infer requests!"
        log.info("Inference Engine version: {}".format(ie_get_version()))

        result = None
        log.info("Creating IE Core Engine...")
        ie = IECore()
        self._configure_plugin(ie)

        log.info("Loading network files")
        net = ie.read_network(
            model=str(resolve_file_path(self.xml)),
            weights=str(resolve_file_path(self.bin)))
        self.network_modifiers.execute(network=net)

        log.info("Loading network to the {} device...".format(self.device))
        exec_net = ie.load_network(network=net, device_name=self.device, num_requests=self.nireq)
        t0 = self._get_thermal_metric(exec_net=exec_net, ie=ie)
        if t0 is not None:
            log.info("{} device thermal metric before infer: {}".format(self.device, t0))

        log.info("Starting async inference for {} requests...".format(self.nireq))
        for i in range(self.nireq):
            input = input_data[i] if self.multi_image else input_data
            exec_net.requests[i].async_infer(input)

        while not all([req.wait(0) == 0 for req in exec_net.requests]):
            pass
        t1 = self._get_thermal_metric(exec_net=exec_net, ie=ie)
        if t1 is not None:
            if isinstance(t1, list):
                diff = [round(t - t0[i], 3) for i, t in enumerate(t1)]
            else:
                diff = round(t1 - t0, 3)
            log.info("{} device thermal metric after infer: {}. \t\nHeating: {}".format(self.device, t1, diff))

        result = [req.outputs for req in exec_net.requests]

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result, -1, -1


class SequenceInferenceAPI20(InferAPI20):
    """Sequence inference engine runner."""
    __action_name__ = "ie_sequence_api_2"

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


class SequenceInference(Infer):
    """Sequence inference engine runner."""
    __action_name__ = "ie_sequence"

    def _infer(self, input_data):
        log.info("Using old API")
        log.info("Inference Engine version: {}".format(ie_get_version()))
        result, load_net_to_plug_time = None, None
        if mem_info_available:
            mem_usage_in_kbytes_before_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        log.info("Creating IE Core Engine...")
        ie = IECore()
        self._configure_plugin(ie)

        log.info("Loading network files")
        if self.model_path:
            net = ie.read_network(model=self.model_path)
        else:
            net = ie.read_network(model=self.xml, weights=self.bin)
        self.network_modifiers.execute(network=net, input_data=input_data)

        log.info("Loading network to the {} device...".format(self.device))
        # Measure time of loading network to plugin
        t_load_to_pl = time.time()
        exec_net = ie.load_network(net, self.device)
        load_net_to_plug_time = time.time() - t_load_to_pl

        t0 = self._get_thermal_metric(exec_net=exec_net, ie=ie)
        if t0 is not None:
            log.info("{} device thermal metric before infer: \n{}".format(self.device, t0, 3))
        # make input_data (dict) a list of frame feed dicts
        for input_layer in input_data:
            frame_size = exec_net.input_info[input_layer].input_data.shape
            input_data[input_layer] = input_data[input_layer].reshape(-1, *frame_size)
        num_frames = max([input_data[key].shape[0] for key in input_data])
        input_data = {key: value if value.shape[0] == num_frames else np.tile(value, num_frames).reshape(num_frames, *(
        list(value.shape)[1:])) for key, value in input_data.items()}
        log.info("Total number of input frames: {}".format(num_frames))
        result = []
        log.info("Starting inference")
        for frame in range(0, num_frames):
            cur_frame_data = {key: value[frame] for key, value in input_data.items()}
            infer_result = exec_net.infer(cur_frame_data)
            result.append(infer_result)
        # make result (list of infer result for each frame) a dict (each layer contains infer result for all frames)
        result = {key: [value[key] for value in result] for key in result[0]}
        result = {key: np.stack(values, axis=0).reshape(num_frames, *(list(values[0].shape[1:]))) for key, values in
                  result.items()}

        t1 = self._get_thermal_metric(exec_net=exec_net, ie=ie)
        if t1 is not None:
            if isinstance(t1, list):
                diff = [round(t - t0[i], 3) for i, t in enumerate(t1)]
            else:
                diff = round(t1 - t0, 3)
            log.info("{} device thermal metric after infer: {}. \t\nHeating: {}".format(self.device, t1, diff))

        if mem_info_available:
            mem_usage_in_kbytes_after_run = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            mem_usage_ie = round((mem_usage_in_kbytes_after_run - mem_usage_in_kbytes_before_run) / 1024)
        else:
            mem_usage_ie = -1

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, mem_usage_ie
