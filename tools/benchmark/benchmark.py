"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import os
from datetime import datetime
from statistics import median
from openvino.inference_engine import IENetwork, IECore, get_version, StatusCode

from .utils.constants import CPU_DEVICE_NAME, MULTI_DEVICE_NAME, GPU_DEVICE_NAME, MYRIAD_DEVICE_NAME, BIN_EXTENSION
from .utils.logging import logger
from .utils.utils import get_duration_seconds, parse_nstreams_value_per_device, parse_devices
from .utils.inputs_filling import get_blob_shape


class Benchmark:
    def __init__(self, device: str, number_infer_requests, number_iterations, duration_seconds, api_type):
        self.device = device
        self.ie = IECore()
        self.nireq = number_infer_requests
        self.niter = number_iterations
        self.duration_seconds = get_duration_seconds(duration_seconds, self.niter, self.device)
        self.api_type = api_type
        self.device_number_streams = {}

    def __del__(self):
        del self.ie

    def add_extension(self, path_to_extension: str=None, path_to_cldnn_config: str=None):
        if GPU_DEVICE_NAME in self.device:
            if path_to_cldnn_config:
                self.ie.set_config({'CONFIG_FILE': path_to_cldnn_config}, GPU_DEVICE_NAME)
                logger.info('GPU extensions is loaded {}'.format(path_to_cldnn_config))
        if CPU_DEVICE_NAME in self.device or MYRIAD_DEVICE_NAME in self.device:
            if path_to_extension:
                self.ie.add_extension(extension_path=path_to_extension, device_name=CPU_DEVICE_NAME)
                logger.info('CPU extensions is loaded {}'.format(path_to_extension))

    def get_version_info(self) -> str:
        logger.info('InferenceEngine:\n{: <9}{:.<24} {}'.format('', 'API version', get_version()))
        version_string = 'Device info\n'
        for device, version in self.ie.get_versions(self.device).items():
            version_string += '{: <9}{}\n'.format('', device)
            version_string += '{: <9}{:.<24}{} {}.{}\n'.format('', version.description, ' version', version.major,
                                                               version.minor)
            version_string += '{: <9}{:.<24} {}\n'.format('', 'Build', version.build_number)
        return version_string

    @staticmethod
    def reshape(ie_network: IENetwork, batch_size: int):
        new_shapes = {}
        for input_layer_name, input_layer in ie_network.inputs.items():
            new_shapes[input_layer_name] = get_blob_shape(input_layer, batch_size)

        if new_shapes:
            logger.info('Resizing network to batch = {}'.format(batch_size))
            ie_network.reshape(new_shapes)

    def set_config(self, number_streams: int, api_type: str = 'async',
                   number_threads: int = None, infer_threads_pinning: int = None):
        devices = parse_devices(self.device)
        self.device_number_streams = parse_nstreams_value_per_device(devices, number_streams)
        for device_name in  self.device_number_streams.keys():
            key = device_name + "_THROUGHPUT_STREAMS"
            supported_config_keys = self.ie.get_metric(device_name, 'SUPPORTED_CONFIG_KEYS')
            if key not in supported_config_keys:
                raise Exception("Device " + device_name + " doesn't support config key '" + key + "'! " +
                                "Please specify -nstreams for correct devices in format  <dev1>:<nstreams1>,<dev2>:<nstreams2>");

        for device in devices:
            if device == CPU_DEVICE_NAME:  # CPU supports few special performance-oriented keys
                # limit threading for CPU portion of inference
                if number_threads:
                    self.ie.set_config({'CPU_THREADS_NUM': str(number_threads)}, device)

                if MULTI_DEVICE_NAME in self.device and GPU_DEVICE_NAME in self.device:
                    self.ie.set_config({'CPU_BIND_THREAD': 'NO'}, CPU_DEVICE_NAME)
                else:
                    # pin threads for CPU portion of inference
                    self.ie.set_config({'CPU_BIND_THREAD': infer_threads_pinning}, device)

                # for CPU execution, more throughput-oriented execution via streams
                # for pure CPU execution, more throughput-oriented execution via streams
                if api_type == 'async':
                    cpu_throughput = {'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
                    if device in self.device_number_streams.keys():
                        cpu_throughput['CPU_THROUGHPUT_STREAMS'] = str(self.device_number_streams.get(device))
                    self.ie.set_config(cpu_throughput, device)
                    self.device_number_streams[device] = self.ie.get_config(device, 'CPU_THROUGHPUT_STREAMS')

            elif device == GPU_DEVICE_NAME:
                if api_type == 'async':
                    gpu_throughput = {'GPU_THROUGHPUT_STREAMS': 'GPU_THROUGHPUT_AUTO'}
                    if device in self.device_number_streams.keys():
                        gpu_throughput['GPU_THROUGHPUT_STREAMS'] = str(self.device_number_streams.get(device))
                    self.ie.set_config(gpu_throughput, device)
                    self.device_number_streams[device] = self.ie.get_config(device, 'GPU_THROUGHPUT_STREAMS')

                if MULTI_DEVICE_NAME in self.device and CPU_DEVICE_NAME in self.device:
                    # multi-device execution with the CPU+GPU performs best with GPU trottling hint,
                    # which releases another CPU thread (that is otherwise used by the GPU driver for active polling)
                    self.ie.set_config({'CLDNN_PLUGIN_THROTTLE': '1'}, device)

            elif device == MYRIAD_DEVICE_NAME:
                self.ie.set_config({'LOG_LEVEL': 'LOG_INFO'}, MYRIAD_DEVICE_NAME)

    def read_network(self, path_to_model: str):
        xml_filename = os.path.abspath(path_to_model)
        head, tail = os.path.splitext(xml_filename)
        bin_filename = os.path.abspath(head + BIN_EXTENSION)

        ie_network = self.ie.read_network(xml_filename, bin_filename)

        input_info = ie_network.inputs

        if not input_info:
            raise AttributeError('No inputs info is provided')

        return ie_network

    def load_network(self, ie_network: IENetwork, perf_counts: bool):
        config = {'PERF_COUNT': ('YES' if perf_counts else 'NO')}

        exe_network = self.ie.load_network(ie_network,
                                           self.device,
                                           config=config,
                                           num_requests=1 if self.api_type == 'sync' else self.nireq or 0)
        # Number of requests
        self.nireq = len(exe_network.requests)
        return exe_network

    def infer(self, exe_network, batch_size, progress_bar=None):
        progress_count = 0
        infer_requests = exe_network.requests

        # warming up - out of scope
        if self.api_type == 'sync':
            infer_requests[0].infer()
        else:
            infer_requests[0].async_infer()
            status = exe_network.wait()
            if status != StatusCode.OK:
                raise Exception("Wait for all requests is failed with status code {}!".format(status))

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
            raise Exception("Wait for all requests is failed with status code {}!".format(status))

        total_duration_sec = (datetime.utcnow() - start_time).total_seconds()
        for infer_request_id in in_fly:
            times.append(infer_requests[infer_request_id].latency)
        times.sort()
        latency_ms = median(times)
        fps = batch_size * 1000 / latency_ms if self.api_type == 'sync' else batch_size * iteration / total_duration_sec
        if progress_bar:
            progress_bar.finish()
        return fps, latency_ms, total_duration_sec, iteration
