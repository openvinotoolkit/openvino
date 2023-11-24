# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from openvino.runtime import Core, get_version, AsyncInferQueue, serialize
import xml.etree.ElementTree as ET

from openvino.tools.hyper_precision_checker.results_process import OV_Result

from .constants import XML_EXTENSION, BIN_EXTENSION
from .logging import logger
from .utils import percentile


class ModelCreator():
    def __init__(self, xml_path):
        self.doc = ET.parse(xml_path)
        self.root = self.doc.getroot()
        self.layers = self.root.find("layers")

    def search_layer(self, layer_name):
        for layer in self.layers.findall("layer"):
            name = layer.attrib.get("name")
            if name == layer_name:
                return layer
        return None

    def update_rt_info(self, rt_info, force_fp32: bool):
        rt_info_attr = None
        for child in rt_info.iter("attribute"):
            if "force_fp32" == child.attr.get("name"):
                rt_info_attr = child
                break
        force_fp32_str = "false"
        if force_fp32:
            force_fp32_str = "true"
        if rt_info_attr is None:
            rt_info_attr = ET.Element(
                "attribute", {"name": "force_fp32", "version": "0", "value": force_fp32_str})
            rt_info.append(rt_info_attr)
        else:
            rt_info_attr.set("value", force_fp32_str)

    def clearup_force_fp32(self):
        for layer in self.layers.iter("layer"):
            rt_info = layer.find("rt_info")
            if rt_info is not None:
                for child in rt_info.iter("attribute"):
                    if "force_fp32" == child.get("name"):
                        rt_info.remove(child)

    def create_new_xml(self, new_xml_path, force_fp32_set):
        self.clearup_force_fp32()
        for iname in force_fp32_set:
            layer = self.search_layer(iname)
            name = layer.get("name")
            rt_info = layer.find("rt_info")
            if rt_info is None:
                rt_info = ET.Element("rt_info")
                layer.append(rt_info)
            self.update_rt_info(rt_info, True)
        self.doc.write(new_xml_path)


class ModelRunner:
    def __init__(self, nstreams):
        self.core = Core()
        self.compiled_model = None
        self.nireq = nstreams
        self.res = None
        self.outputs = []
        self.requests = None

    def __del__(self):
        del self.core
        if self.compiled_model is not None:
            del self.compiled_model

    def save_model(self, path_to_save: str):
        serialize(self.model, xml_path=path_to_save + "/serialized.xml",
                  bin_path=path_to_save + "/serialized.bin")

    def print_version_info(self) -> None:
        version = get_version()
        logger.info('OpenVINO:')
        logger.info(f"{'Build ':.<39} {version}")
        logger.info("")

        logger.info("Device info:")
        for device, version in self.core.get_versions(self.device).items():
            logger.info(f"{device}")
            logger.info(f"{'Build ':.<39} {version.build_number}")

        logger.info("")
        logger.info("")

    def read_model(self, path_to_model_xml: str, path_to_model_bin: str):
        model_filename = os.path.abspath(path_to_model_xml)
        if path_to_model_bin is None:
            head, ext = os.path.splitext(model_filename)
            weights_filename = os.path.abspath(
                head + BIN_EXTENSION) if ext == XML_EXTENSION else ""
        else:
            weights_filename = path_to_model_bin
        self.model = self.core.read_model(model_filename, weights_filename)
        output_size = self.model.get_output_size()
        # self.outputs.append([0,"abc"])
        for i in range(0, output_size):
            self.outputs.append(
                [i, self.model.get_output_op(i).get_friendly_name()])
        self.res = OV_Result(self.outputs)
        return

    def compile_model(self, data_type):
        config = {}
        supported_properties = self.core.get_property(
            "CPU", 'SUPPORTED_PROPERTIES')
        config['NUM_STREAMS'] = self.nireq
        config['AFFINITY'] = 'CORE'
        config['INFERENCE_NUM_THREADS'] = "0"  # str(stream_num) #"0"
        config['PERF_COUNT'] = 'YES'
        config['INFERENCE_PRECISION_HINT'] = data_type  # 'bf16'#'f32'
        config['PERFORMANCE_HINT'] = 'THROUGHPUT'  # 'THROUGHPUT' #"LATENCY"
        config['CPU_THREADS_NUM'] = "0"
        config['CPU_BIND_THREAD'] = 'YES'  # 'YES'#'NUMA' #'HYBRID_AWARE'
        self.compiled_model = self.core.compile_model(
            self.model, 'CPU', config)

    def create_infer_requests(self):
        self.requests = AsyncInferQueue(self.compiled_model, self.nireq)
        self.nireq = len(self.requests)
        self.requests.set_callback(self.res.completion_callback)
        return self.requests

    def first_infer(self, data_queue):
        idle_id = self.requests.get_idle_request_id()
        data, index = data_queue.get_first_input()
        self.requests[idle_id].set_input_tensors(data)
        self.requests.start_async()
        self.requests.wait_all()
        return self.requests[idle_id].latency

    def async_inference(self, data_queue):
        processed_frames = 0
        exec_time = 0
        iteration = 0
        times = []
        start_time = datetime.utcnow()
        in_fly = set()
        while data_queue.has_next_input():
            processed_frames += 1
            idle_id = self.requests.get_idle_request_id()
            if idle_id in in_fly:
                times.append(self.requests[idle_id].latency)
            else:
                in_fly.add(idle_id)
            data, index = data_queue.get_next_input()
            self.requests[idle_id].set_input_tensors(data)
            self.requests.start_async(userdata=index)
            iteration += 1

        exec_time = (datetime.utcnow() - start_time).total_seconds()
        self.requests.wait_all()
        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()

        for infer_request_id in in_fly:
            times.append(self.requests[infer_request_id].latency)

        return sorted(times), total_duration_sec, processed_frames, iteration

    def main_loop(self, data_queue, latency_percentile):
        times, total_duration_sec, processed_frames, iteration = self.async_inference(
            data_queue)
        fps = processed_frames / total_duration_sec

        median_latency_ms = percentile(times, latency_percentile)
        avg_latency_ms = sum(times) / len(times)
        min_latency_ms = times[0]
        max_latency_ms = times[-1]

        return self.res, fps, median_latency_ms, avg_latency_ms, min_latency_ms, max_latency_ms, total_duration_sec, iteration
