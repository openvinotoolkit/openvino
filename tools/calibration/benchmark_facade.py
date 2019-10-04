"""
Copyright (C) 2019 Intel Corporation

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

from ..benchmark.benchmark import Benchmark
from ..benchmark.utils.progress_bar import ProgressBar
from ..benchmark.utils.utils import read_network
from ..benchmark.utils.inputs_filling import get_inputs
from ..benchmark.utils.infer_request_wrap import InferRequestsQueue


class BenchmarkResult:
    def __init__(self, latency):
        self._latency = latency

    @property
    def latency(self) -> float:
        return self._latency

class BenchmarkFacade:
    def __init__(self, device, batch_size, benchmark_iterations_count, cpu_extension):
        self._benchmark = Benchmark(device, 1, benchmark_iterations_count, None, "sync")
        self._benchmark.add_extension(cpu_extension)
        self._progress_bar_total_count = benchmark_iterations_count \
            if benchmark_iterations_count and not self._benchmark.duration_seconds else 10000
        self._progress_bar = ProgressBar(self._progress_bar_total_count)
        self._batch_size = batch_size

    def run(self, path_to_model):
        ie_network = read_network(path_to_model)
        exe_network = self._benchmark.load_network(ie_network, True, 1)
        request_queue = InferRequestsQueue(exe_network.requests)
        requests_input_data = get_inputs("", self._batch_size, ie_network.inputs, exe_network.requests)
        fps, latency, fp32_total_duration, fp32_iter = self._benchmark.infer(
            request_queue, requests_input_data, self._batch_size, self._progress_bar)

        return BenchmarkResult(latency)
