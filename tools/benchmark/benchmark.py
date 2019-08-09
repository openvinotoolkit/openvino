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

import numpy
import datetime

import openvino.inference_engine as ie

from ..accuracy_checker.accuracy_checker.config import ConfigReader
from ..accuracy_checker.accuracy_checker.evaluators.model_evaluator import ModelEvaluator
from ..accuracy_checker.accuracy_checker.progress_reporters import PrintProgressReporter, TQDMReporter

from ..network import Network

from .configuration import Configuration
from .logging import info


class BenchmarkCallback:
    def __init__(self, configuration: Configuration, network: Network=None, iterations_count:int=1000):
        self._latency = None
        self._configuration = configuration
        self._network = network
        self._iterations_count = iterations_count if iterations_count else 1000

    def output_callback(self, value, latency = None):
        pass


    def benchmark_callback(self, network_inputs_data):
        latencies = list()

        if self._network:
            ie_network = self._network.ie_network
        else:
            ie_network = ie.IENetwork(self._configuration.model, self._configuration.weights)

        do_reshape = False
        for name in ie_network.inputs.keys():
            if name in network_inputs_data and \
                    tuple(ie_network.inputs[name].shape) != network_inputs_data[name].shape:
                do_reshape = True
                break

        if do_reshape:
            new_shapes = {layer_name: data.shape for layer_name, data in network_inputs_data.items()}
            ie_network.reshape(new_shapes)

        plugin = ie.IEPlugin(self._configuration.device)
        if self._configuration.cpu_extension:
            plugin.add_cpu_extension(self._configuration.cpu_extension)
        exec_network = plugin.load(ie_network)

        # warming up
        exec_network.infer(network_inputs_data)

        for i in range(self._iterations_count):
            start = datetime.datetime.now()
            exec_network.infer(network_inputs_data)
            latencies.append((datetime.datetime.now() - start).microseconds)
        self._latency = numpy.mean(latencies) / 1000000.0

        del ie_network
        del exec_network
        del plugin


    @property
    def latency(self) -> float:
        return self._latency


class BenchmarkResult:
    def __init__(self, latency):
        self._latency = latency

    @property
    def latency(self) -> float:
        return self._latency


class InferOptions:
    def __init__(self, iterations_count=1000):
        self._iterations_count = iterations_count

    @property
    def iterations_count(self) -> int:
        return self._iterations_count


class Benchmark:
    def __init__(self, configuration: Configuration):
        if configuration is None:
            raise ValueError("configuration is None")

        self._configuration = configuration
        pass

    def run(
        self,
        network: Network = None,
        statistics=None,
        quantization_levels=None,
        iterations_count:int = 1000) -> BenchmarkResult:

        model = self._configuration.config['models'][0]
        launcher_config = model['launchers'][0]
        dataset_config = model['datasets'][0]

        model_evaluator = ModelEvaluator.from_configs(launcher_config, dataset_config)
        try:
            if network:
                del model_evaluator.launcher.network
                del model_evaluator.launcher.exec_network
                model_evaluator.launcher.network = network.ie_network
                model_evaluator.launcher.exec_network = model_evaluator.launcher.plugin.load(network.ie_network)

            ie_network = model_evaluator.launcher.network

            if statistics:
                network_stats = {}
                for layer_name, node_statistic in statistics.items():
                    network_stats[layer_name] = ie.LayerStats(
                        min=tuple(node_statistic.min_outputs),
                        max=tuple(node_statistic.max_outputs))
                ie_network.stats.update(network_stats)

            if quantization_levels:
                for layer_name, value in quantization_levels.items():
                    params = ie_network.layers[layer_name].params
                    params["quantization_level"] = value
                    ie_network.layers[layer_name].params = params

            if model_evaluator.dataset.size != 1:
                info("only one first image is used from dataset annotation to perform benchmark")
                model_evaluator.dataset.size = 1

            process_dataset_callback = BenchmarkCallback(
                configuration=self._configuration,
                network=network,
                iterations_count=iterations_count)

            model_evaluator.process_dataset(
                None,
                progress_reporter=None,
                output_callback=process_dataset_callback.output_callback,
                benchmark=process_dataset_callback.benchmark_callback)

            if len(model_evaluator.launcher.exec_network.requests) != 1:
                raise ValueError("unexpected network requests count")

            latency = process_dataset_callback.latency
        finally:
            model_evaluator.release()

        return BenchmarkResult(latency)
