"""Inference engine runners."""
import logging as log
# pylint:disable=import-error
import sys
from pprint import pformat
from timeit import default_timer as timer

from openvino.inference_engine import IECore, get_version as ie_get_version
from openvino.runtime import Core

from utils.kaldi_utils import get_quantization_scale_factors
from e2e_oss.utils.path_utils import resolve_file_path
# import local modules:
from .common_inference import Infer

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


class SpeechKaldiInfer(Infer):
    """ Inference engine runner for speech Kaldi networks """
    __action_name__ = "ie_speech_kaldi"

    def __init__(self, config):
        super().__init__(config)
        if self.plugin_cfg_target_device == "GNA":
            default_qb = 16
            default_device_mode = "GNA_SW_EXACT"
            self.plugin_cfg = {
                "GNA_PRECISION": "I{}".format(config.get("qb", default_qb) or default_qb),
                "GNA_DEVICE_MODE": "{}".format(config.get("device_mode", default_device_mode) or default_device_mode),
                "GNA_COMPACT_MODE": "YES",
            }

    def _infer(self, input_data):
        log.info("Inference Engine version: {}".format(ie_get_version()))
        log.info("Creating IE Core Engine...")
        ie = IECore()
        self._configure_plugin(ie)

        log.info("Loading network files")
        if self.model_path:
            net = ie.read_network(model=self.model_path)
        else:
            net = ie.read_network(
                model=str(resolve_file_path(self.xml)),
                weights=str(resolve_file_path(self.bin)))

        self.network_modifiers.execute(network=net)

        if "GNA" in self.plugin_cfg_target_device:
            first_utterance = next(iter(input_data))
            scale_factors = get_quantization_scale_factors(input_data[first_utterance])
            for i, input_name in enumerate(scale_factors):
                scale_factor_key = "GNA_SCALE_FACTOR_{}".format(i)
                ie.set_property('GNA', {scale_factor_key: str(scale_factors[input_name])})
                log.info("Scale factor was set for input '{}': {}".format(input_name,
                                                                          ie.get_property('GNA', scale_factor_key)))

        if self.plugin_cfg:
            log.info("Setting config to the {} plugin. \nConfig:\n{}".format(self.plugin_cfg_target_device,
                                                                             pformat(self.plugin_cfg)))
            ie.set_property(self.plugin_cfg_target_device, self.plugin_cfg)

        log.info("Loading network to the {} device...".format(self.device))
        start = timer()
        exec_net = ie.load_network(network=net, device_name=self.device)
        load_net_to_plug_time = timer() - start
        log.info("Network loading to plugin time: {}".format(load_net_to_plug_time))

        result = {}
        for utterance, frames in input_data.items():
            log.info("Utterance {}:".format(utterance))
            log.info("Frames in utterance: {}".format(len(frames)))
            utterance_results = []
            frame_infer_times = []
            for frame in frames:
                start = timer()
                frame_result = exec_net.infer(frame)
                end = timer()
                frame_infer_time = (end - start) * 1000
                frame_infer_times.append(frame_infer_time)
                utterance_results.append(frame_result)
            result.update({utterance: utterance_results})
            log.info("Average infer time per frame: {} ms"
                     .format(sum(frame_infer_times) / len(frame_infer_times)))
            log.info("End of utterance: {}\n".format(utterance))

        if "exec_net" in locals():
            del exec_net
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, -1


class SpeechKaldiInferAPI2(Infer):
    """ Inference engine runner for speech Kaldi networks """
    __action_name__ = "ie_speech_kaldi_api_2"

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
        log.info("Using API v2.0")
        log.info("Inference Engine version: {}".format(ie_get_version()))
        log.info("Creating IE Core Engine...")
        ie = Core()
        self._configure_plugin(ie)

        log.info("Loading network files")
        if self.model_path:
            self.ov_model = ie.read_model(model=self.model_path)
        if self.xml:
            self.ov_model = ie.read_model(model=str(resolve_file_path(self.xml)))

        self.network_modifiers.execute(network=self.ov_model)

        if "GNA" in self.plugin_cfg_target_device:
            first_utterance = next(iter(input_data))
            scale_factors = get_quantization_scale_factors(input_data[first_utterance])
            for i, input_name in enumerate(scale_factors):
                scale_factor_key = "GNA_SCALE_FACTOR_{}".format(i)
                ie.set_property('GNA', {scale_factor_key: str(scale_factors[input_name])})
                log.info("Scale factor was set for input '{}': {}".format(input_name,
                                                                          ie.get_property('GNA', scale_factor_key)))

        if self.plugin_cfg:
            log.info("Setting config to the {} plugin. \nConfig:\n{}".format(self.plugin_cfg_target_device,
                                                                             pformat(self.plugin_cfg)))
            ie.set_property(self.plugin_cfg_target_device, self.plugin_cfg)

        log.info("Loading network to the {} device...".format(self.device))
        start = timer()
        compiled_model = ie.compile_model(self.ov_model, self.device)
        load_net_to_plug_time = timer() - start
        log.info("Network loading to plugin time: {}".format(load_net_to_plug_time))

        for input_obj in compiled_model.inputs:
            assert input_obj.names, "Input {} has no names".format(input_obj)

        result = {}
        for utterance, frames in input_data.items():
            log.info("Utterance {}:".format(utterance))
            log.info("Frames in utterance: {}".format(len(frames)))
            utterance_results = []
            frame_infer_times = []
            request = compiled_model.create_infer_request()
            for state in request.query_state():
                state.reset()
            for frame in frames:
                start = timer()
                request_result = request.infer(frame)
                end = timer()

                frame_infer_time = (end - start) * 1000
                frame_infer_times.append(frame_infer_time)

                frame_result = {}
                for out_obj, out_tensor in request_result.items():
                    # all input and output tensors have to be named
                    assert out_obj.names, "Output tensor {} has no names".format(out_obj)

                    tensor_name = out_obj.get_any_name()
                    frame_result[tensor_name] = out_tensor

                utterance_results.append(frame_result)

            result.update({utterance: utterance_results})
            log.info("Average infer time per frame: {} ms"
                     .format(sum(frame_infer_times) / len(frame_infer_times)))
            log.info("End of utterance: {}\n".format(utterance))

        if "exec_net" in locals():
            del compiled_model
        if "ie" in locals():
            del ie

        return result, load_net_to_plug_time, -1
