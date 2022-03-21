# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from math import ceil
from typing import Union
from openvino.runtime import Core, get_version, AsyncInferQueue

from .utils.constants import MULTI_DEVICE_NAME, HETERO_DEVICE_NAME, CPU_DEVICE_NAME, GPU_DEVICE_NAME, XML_EXTENSION, BIN_EXTENSION
from .utils.logging import logger
from .utils.utils import get_duration_seconds
from .utils.statistics_report import StatisticsReport

def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]

class Benchmark:
    def __init__(self, device: str, number_infer_requests: int = 0, number_iterations: int = None,
                 duration_seconds: int = None, api_type: str = 'async', inference_only = None):
        self.device = device
        self.core = Core()
        self.nireq = number_infer_requests if api_type == 'async' else 1
        self.niter = number_iterations
        self.duration_seconds = get_duration_seconds(duration_seconds, self.niter, self.device)
        self.api_type = api_type
        self.inference_only = inference_only
        self.latency_groups = []

    def __del__(self):
        del self.core

    def add_extension(self, path_to_extension: str=None, path_to_cldnn_config: str=None):
        if path_to_cldnn_config:
            self.core.set_property(GPU_DEVICE_NAME, {'CONFIG_FILE': path_to_cldnn_config})
            logger.info(f'GPU extensions is loaded {path_to_cldnn_config}')

        if path_to_extension:
            self.core.add_extension(extension_path=path_to_extension)
            logger.info(f'CPU extensions is loaded {path_to_extension}')

    def get_version_info(self) -> str:
        logger.info(f"OpenVINO:\n{'': <9}{'API version':.<24} {get_version()}")
        version_string = 'Device info\n'
        for device, version in self.core.get_versions(self.device).items():
            version_string += f"{'': <9}{device}\n"
            version_string += f"{'': <9}{version.description:.<24}{' version'} {version.major}.{version.minor}\n"
            version_string += f"{'': <9}{'Build':.<24} {version.build_number}\n"
        return version_string

    def set_config(self, config = {}):
        for device in config.keys():
            self.core.set_property(device, config[device])

    def set_cache_dir(self, cache_dir: str):
        self.core.set_property({'CACHE_DIR': cache_dir})

    def read_model(self, path_to_model: str):
        model_filename = os.path.abspath(path_to_model)
        head, ext = os.path.splitext(model_filename)
        weights_filename = os.path.abspath(head + BIN_EXTENSION) if ext == XML_EXTENSION else ""
        return self.core.read_model(model_filename, weights_filename)

    def create_infer_requests(self, compiled_model):
        if self.api_type == 'sync':
            requests = [compiled_model.create_infer_request()]
        else:
            requests = AsyncInferQueue(compiled_model, self.nireq)
            self.nireq = len(requests)
        return requests

    def first_infer(self, requests):
        if self.api_type == 'sync':
            requests[0].infer()
            return requests[0].latency
        else:
            id = requests.get_idle_request_id()
            requests.start_async()
            requests.wait_all()
            return requests[id].latency

    def update_progress_bar(self, progress_bar, exec_time, progress_count):
        if self.duration_seconds:
            # calculate how many progress intervals are covered by current iteration.
            # depends on the current iteration time and time of each progress interval.
            # Previously covered progress intervals must be skipped.
            progress_interval_time = self.duration_seconds / progress_bar.total_num
            new_progress = int(exec_time / progress_interval_time - progress_count)
            progress_bar.add_progress(new_progress)
            progress_count += new_progress
        elif self.niter:
            progress_bar.add_progress(1)
        return progress_count

    def sync_inference(self, request, data_queue, progress_bar):
        progress_count = 0
        exec_time = 0
        iteration = 0
        times = []
        start_time = datetime.utcnow()
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds):
            if self.inference_only == False:
                request.set_input_tensors(data_queue.get_next_input())
            request.infer()
            times.append(request.latency)
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()

            if progress_bar:
                progress_count = self.update_progress_bar(progress_bar, exec_time, progress_count)

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        return sorted(times), total_duration_sec, iteration

    def async_inference_only(self, infer_queue, progress_bar):
        progress_count = 0
        exec_time = 0
        iteration = 0
        times = []
        in_fly = set()
        start_time = datetime.utcnow()
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds) or \
              (iteration % self.nireq):
            idle_id = infer_queue.get_idle_request_id()
            if idle_id in in_fly:
                times.append(infer_queue[idle_id].latency)
            else:
                in_fly.add(idle_id)
            infer_queue.start_async()
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()

            if progress_bar:
                progress_count = self.update_progress_bar(progress_bar, exec_time, progress_count)

        infer_queue.wait_all()
        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_queue[infer_request_id].latency)
        return sorted(times), total_duration_sec, iteration

    def async_inference_full_mode(self, infer_queue, data_queue, progress_bar, pcseq):
        progress_count = 0
        processed_frames = 0
        exec_time = 0
        iteration = 0
        times = []
        num_groups = len(self.latency_groups)
        in_fly = set()
        start_time = datetime.utcnow()
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds) or \
              (iteration % num_groups):
            processed_frames += data_queue.get_next_batch_size()
            idle_id = infer_queue.get_idle_request_id()
            if idle_id in in_fly:
                times.append(infer_queue[idle_id].latency)
                if pcseq:
                    self.latency_groups[infer_queue.userdata[idle_id]].times.append(infer_queue[idle_id].latency)
            else:
                in_fly.add(idle_id)
            group_id = data_queue.current_group_id
            infer_queue[idle_id].set_input_tensors(data_queue.get_next_input())
            infer_queue.start_async(userdata=group_id)
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()

            if progress_bar:
                progress_count = self.update_progress_bar(progress_bar, exec_time, progress_count)

        infer_queue.wait_all()
        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_queue[infer_request_id].latency)
        return sorted(times), total_duration_sec, processed_frames, iteration

    def main_loop(self, requests, data_queue, batch_size, latency_percentile, progress_bar, pcseq):
        if self.api_type == 'sync':
            times, total_duration_sec, iteration = self.sync_inference(requests[0], data_queue, progress_bar)
        elif self.inference_only:
            times, total_duration_sec, iteration = self.async_inference_only(requests, progress_bar)
            fps = len(batch_size) * iteration / total_duration_sec
        else:
            times, total_duration_sec, processed_frames, iteration = self.async_inference_full_mode(requests, data_queue, progress_bar, pcseq)
            fps = processed_frames / total_duration_sec

        median_latency_ms = percentile(times, latency_percentile)
        avg_latency_ms = sum(times) / len(times)
        min_latency_ms = times[0]
        max_latency_ms = times[-1]

        if self.api_type == 'sync':
            fps = len(batch_size) * 1000 / median_latency_ms

        if pcseq:
            for group in self.latency_groups:
                if group.times:
                    group.times.sort()
                    group.avg = sum(group.times) / len(group.times)
                    group.min = group.times[0]
                    group.max = group.times[-1]

        if progress_bar:
            progress_bar.finish()
        return fps, median_latency_ms, avg_latency_ms, min_latency_ms, max_latency_ms, total_duration_sec, iteration
