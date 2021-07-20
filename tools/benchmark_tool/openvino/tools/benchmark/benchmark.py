# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from statistics import median
from openvino.inference_engine import IENetwork, IECore, get_version, StatusCode

from .utils.constants import MULTI_DEVICE_NAME, HETERO_DEVICE_NAME, CPU_DEVICE_NAME, GPU_DEVICE_NAME, XML_EXTENSION, BIN_EXTENSION
from .utils.logging import logger
from .utils.utils import get_duration_seconds
from .utils.statistics_report import StatisticsReport

class Benchmark:
    def __init__(self, device: str, number_infer_requests: int = None, number_iterations: int = None,
                 duration_seconds: int = None, api_type: str = 'async'):
        self.device = device
        self.ie = IECore()
        self.nireq = number_infer_requests
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
            self.ie.add_extension(extension_path=path_to_extension, device_name=CPU_DEVICE_NAME)
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

    def read_network(self, path_to_model: str):
        model_filename = os.path.abspath(path_to_model)
        head, ext = os.path.splitext(model_filename)
        weights_filename = os.path.abspath(head + BIN_EXTENSION) if ext == XML_EXTENSION else ""
        ie_network = self.ie.read_network(model_filename, weights_filename)
        return ie_network

    def load_network(self, ie_network: IENetwork, config = {}):
        exe_network = self.ie.load_network(ie_network,
                                           self.device,
                                           config=config,
                                           num_requests=1 if self.api_type == 'sync' else self.nireq or 0)
        # Number of requests
        self.nireq = len(exe_network.requests)

        return exe_network

    def load_network_from_file(self, path_to_model: str, config = {}):
        exe_network = self.ie.load_network(path_to_model,
                                           self.device,
                                           config=config,
                                           num_requests=1 if self.api_type == 'sync' else self.nireq or 0)
        # Number of requests
        self.nireq = len(exe_network.requests)

        return exe_network

    def import_network(self, path_to_file : str, config = {}):
        exe_network = self.ie.import_network(model_file=path_to_file,
                                             device_name=self.device,
                                             config=config,
                                             num_requests=1 if self.api_type == 'sync' else self.nireq or 0)
        # Number of requests
        self.nireq = len(exe_network.requests)
        return exe_network

    def first_infer(self, exe_network):
        infer_request = exe_network.requests[0]

        # warming up - out of scope
        if self.api_type == 'sync':
            infer_request.infer()
        else:
            infer_request.async_infer()
            status = infer_request.wait()
            if status != StatusCode.OK:
                raise Exception(f"Wait for all requests is failed with status code {status}!")
        return infer_request.latency

    def infer(self, exe_network, batch_size, progress_bar=None):
        progress_count = 0
        infer_requests = exe_network.requests

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
                infer_requests[0].infer()
                times.append(infer_requests[0].latency)
            else:
                infer_request_id = exe_network.get_idle_request_id()
                if infer_request_id < 0:
                    status = exe_network.wait(num_requests=1)
                    if status != StatusCode.OK:
                        raise Exception("Wait for idle request failed!")
                    infer_request_id = exe_network.get_idle_request_id()
                    if infer_request_id < 0:
                        raise Exception("Invalid request id!")
                if infer_request_id in in_fly:
                    times.append(infer_requests[infer_request_id].latency)
                else:
                    in_fly.add(infer_request_id)
                infer_requests[infer_request_id].async_infer()
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
        status = exe_network.wait()
        if status != StatusCode.OK:
            raise Exception(f"Wait for all requests is failed with status code {status}!")

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_requests[infer_request_id].latency)
        times.sort()
        latency_ms = median(times)
        fps = batch_size * 1000 / latency_ms if self.api_type == 'sync' else batch_size * iteration / total_duration_sec
        if progress_bar:
            progress_bar.finish()
        return fps, latency_ms, total_duration_sec, iteration
