"""
Copyright (C) 2018-2019 Intel Corporation

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

import openvino.inference_engine as ie

from ..infer_raw_results import InferRawResults
from ..aggregated_statistics import AggregatedStatistics

class CollectResultsCallback:
    def __init__(
        self,
        network: ie.IENetwork,
        exec_network: ie.ExecutableNetwork,
        collect_resuls: bool = True,
        collect_layers: set = None,
        collect_aggregated_statistics: bool = True,
        iterations_count: int = 1,
        dataset_size: int = 1):

        if not network:
            raise ValueError("network is not specified")
        if not exec_network:
            raise ValueError("exec_network is not specified")

        self._network = network
        self._exec_network = exec_network
        self._aggregated_statistics = None
        self._iterations_count = iterations_count
        self._dataset_size = dataset_size
        self._collect_results = collect_resuls
        self._collect_layers = collect_layers
        self._collect_aggregated_statistics = collect_aggregated_statistics
        self._infer_raw_results = InferRawResults() if collect_resuls else None
        self._latencies = list()

    def callback(self, value, latency=None, **kwargs):
        network = kwargs.get('network')
        exec_network = kwargs.get('exec_network')
        if not network or not exec_network:
            network = self._network
            exec_network = self._exec_network
        if self._collect_aggregated_statistics:
            if not self._aggregated_statistics:
                self._aggregated_statistics = AggregatedStatistics(
                    iterations_count = self._iterations_count,
                    dataset_size = self._dataset_size)
            self._aggregated_statistics.add(network, exec_network, value)

        if self._collect_results:
            if self._collect_layers:
                collect_value = dict()
                for layer_name in value:
                    if layer_name in self._collect_layers:
                        collect_value[layer_name] = value[layer_name]
                self._infer_raw_results.add(collect_value)
            else:
                self._infer_raw_results.add(value)

        if latency:
            self._latencies.append(latency)

    @property
    def aggregated_statistics(self) -> AggregatedStatistics:
        return self._aggregated_statistics

    @property
    def infer_raw_result(self) -> InferRawResults:
        return self._infer_raw_results

    @property
    def latencies(self) -> list:
        return self._latencies

    def release(self):
        if self._aggregated_statistics:
            self._aggregated_statistics.release()
        if self._infer_raw_results:
            self._infer_raw_results.release()

    def get_accuracy_drop(self):
        return None
