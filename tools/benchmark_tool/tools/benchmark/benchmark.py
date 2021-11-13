# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from math import ceil
from typing import Union
from openvino import Core, Function, get_version

from .utils.constants import MULTI_DEVICE_NAME, HETERO_DEVICE_NAME, CPU_DEVICE_NAME, GPU_DEVICE_NAME, XML_EXTENSION, BIN_EXTENSION
from .utils.logging import logger
from .utils.utils import get_duration_seconds
from .utils.statistics_report import StatisticsReport

def percentile(values, percent):
    return values[ceil(len(values) * percent / 100) - 1]

class Benchmark:
    def __init__(self, device: str, number_infer_requests: int = 0, number_iterations: int = None,
                 duration_seconds: int = None, api_type: str = 'async'):
        self.device = device
        self.ie = Core()
        self.nireq = number_infer_requests if api_type == 'async' else 1
        self.niter = number_iterations
        self.duration_seconds = get_duration_seconds(duration_seconds, self.niter, self.device)
        self.api_type = api_type

    def __del__(self):
        del self.ie

    def add_extension(self, path_to_extension: str=None, path_to_cldnn_config: str=None):
        if path_to_cldnn_config:
            self.ie.set_config({'CONFIG_FILE': path_to_cldnn_config}, GPU_DEVICE_NAME)
            logger.info(f'GPU extensions is loaded {path_to_cldnn_config}')

        if path_to_extension:
            self.ie.add_extension(extension_path=path_to_extension)
            logger.info(f'CPU extensions is loaded {path_to_extension}')

    def get_version_info(self) -> str:
        logger.info(f"InferenceEngine:\n{'': <9}{'API version':.<24} {get_version()}")
        version_string = 'Device info\n'
        for device, version in self.ie.get_versions(self.device).items():
            version_string += f"{'': <9}{device}\n"
            version_string += f"{'': <9}{version.description:.<24}{' version'} {version.major}.{version.minor}\n"
            version_string += f"{'': <9}{'Build':.<24} {version.build_number}\n"
        return version_string

    def set_config(self, config = {}):
        for device in config.keys():
            self.ie.set_config(config[device], device)

    def set_cache_dir(self, cache_dir: str):
        self.ie.set_config({'CACHE_DIR': cache_dir}, '')

    def read_model(self, path_to_model: str):
        model_filename = os.path.abspath(path_to_model)
        head, ext = os.path.splitext(model_filename)
        weights_filename = os.path.abspath(head + BIN_EXTENSION) if ext == XML_EXTENSION else ""
        return self.ie.read_model(model_filename, weights_filename)

    def first_infer(self, infer_queue):
        infer_request = infer_queue[0]

        # warming up - out of scope
        if self.api_type == 'sync':
            infer_request.infer()
        else:
            infer_request.async_infer()
            infer_request.wait()
        return infer_request.latency

    def infer(self, infer_queue, batch_size, latency_percentile, progress_bar=None):
        progress_count = 0

        start_time = datetime.utcnow()
        exec_time = 0
        iteration = 0

        times = []
        in_fly = set()
        # Start inference & calculate performance
        # to align number if iterations to guarantee that last infer requests are executed in the same conditions **/
        while (self.niter and iteration < self.niter) or \
              (self.duration_seconds and exec_time < self.duration_seconds) or \
              (self.api_type == 'async' and iteration % self.nireq):
            if self.api_type == 'sync':
                infer_queue[0].infer()
                times.append(infer_queue[0].latency)
            else:
                idle_id = infer_queue.get_idle_request_id()
                if idle_id in in_fly:
                    times.append(infer_queue[idle_id].latency)
                else:
                    in_fly.add(idle_id)
                infer_queue.start_async()
            iteration += 1

            exec_time = (datetime.utcnow() - start_time).total_seconds()

            if progress_bar:
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

        # wait the latest inference executions
        infer_queue.wait_all()

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_queue[infer_request_id].latency)
        times.sort()
        latency_ms = percentile(times, latency_percentile)
        fps = batch_size * 1000 / latency_ms if self.api_type == 'sync' else batch_size * iteration / total_duration_sec
        if progress_bar:
            progress_bar.finish()
        return fps, latency_ms, total_duration_sec, iteration
