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
from ..aggregated_statistics import AggregatedStatistics
from ..infer_raw_results import InferRawResults
from ...utils.network_info import NetworkInfo
from ..logging import info, debug
from ...network import Network
from ..calibration_configuration import CalibrationConfiguration
from ..single_layer_network import SingleLayerNetwork
from ..layer_accuracy_drop_info import LayerAccuracyDropInfo
import numpy as np
import threading
from ..nrmsd import compare_nrmsd
import multiprocessing

class CalculateAccuracyCallback:
    def __init__(
        self,
        network: ie.IENetwork,
        exec_network: ie.ExecutableNetwork,
        collect_layers: set,
        configuration: CalibrationConfiguration,
        per_layer_statistics: dict,
        normalizer,
        ignore_layer_names=None):

        if not network:
            raise ValueError("network is not specified")
        if not exec_network:
            raise ValueError("exec_network is not specified")
        if not configuration:
            raise ValueError("configuration is not specified")
        if not collect_layers:
            raise ValueError("layers to collect is not specified")
        if not per_layer_statistics:
            raise ValueError("statistics is not specified")
        if not normalizer:
            raise ValueError("normalizer is not specified")

        self._network = network
        self._exec_network = exec_network
        self._collect_layers = collect_layers
        self._configuration = configuration
        self._per_layer_statistics = per_layer_statistics
        self._normalizer = normalizer
        self._ignore_layer_names = ignore_layer_names

        self._accuracy_drop_dict = None
        self._single_layer_networks = np.array([])
        self._layers_to_return_to_fp32 = np.array([])
        self._callback_preparation = False
        self._active_threads = list()

    def accuracy_drop_for_layer(self, accuracy_drop_for_every_picture : np.array) -> np.float64:
        sum = np.float64(0)
        for i in accuracy_drop_for_every_picture:
            sum += i
        return sum / accuracy_drop_for_every_picture.size

    def get_accuracy_drop(self) -> np.array:
        for thread in self._active_threads:
            thread.join()
        self._active_threads.clear()

        accuracy_drop = []
        single_layer_network_names = [net.layer_name for net in self._single_layer_networks]
        for layer_name, accuracy_drop_of_this_layer in self._accuracy_drop_dict.items():
            if layer_name in single_layer_network_names and accuracy_drop_of_this_layer.size != 0:
                accuracy_drop.append(LayerAccuracyDropInfo(
                    layer_name=layer_name,
                    value=self.accuracy_drop_for_layer(accuracy_drop_of_this_layer),
                    precision=self._network.layers[layer_name].precision))

        accuracy_drop.sort(key=lambda accuracy_drop: accuracy_drop.value, reverse=True)
        return accuracy_drop

    def calculate_accuracy_drop(self,
                                infer_raw_results: InferRawResults):
        threads_count = multiprocessing.cpu_count()
        ntw_count = self._single_layer_networks.size
        if ntw_count == 0:
            return
        work_per_thread = int(ntw_count / threads_count)
        if work_per_thread == 0:
            work_per_thread = 1
        begin = 0
        end = work_per_thread

        while ntw_count > 0:
            if end > self._single_layer_networks.size:
                end = self._single_layer_networks.size
            thread = threading.Thread(target=self.thread_loop,
                                      args=(infer_raw_results, begin, end,))
            thread.start()
            self._active_threads.append(thread)

            ntw_count -= work_per_thread
            begin = end
            end += work_per_thread

    def thread_loop(self, infer_raw_results, begin: int, end: int):
        for i in range(begin, end):
            self.calculate_accuracy_drop_for_layer(infer_raw_results, i)

    def calculate_accuracy_drop_for_layer(self,
                                          infer_raw_results,
                                          index: int):
        for j in self.calculate_accuracy_drop_for_batch(infer_raw_results,
                                                        self._single_layer_networks[index]):
            self._accuracy_drop_dict[self._layers_to_return_to_fp32[index].name] = \
                np.append(self._accuracy_drop_dict[self._layers_to_return_to_fp32[index].name], j)


    def calculate_accuracy_drop_for_batch(self,
                                        infer_raw_results,
                                        single_layer_network: SingleLayerNetwork) -> np.array:
        assert infer_raw_results is not None, "infer_raw_results should be set"

        accuracy_drop_list = np.array([])

        for raw_data in infer_raw_results:
            if single_layer_network.input_layer_name not in raw_data:
                continue
            input_layer_data = raw_data[single_layer_network.input_layer_name]

            if tuple(single_layer_network._network.inputs[single_layer_network.input_layer_name].shape) != input_layer_data.shape:
                new_shapes = {single_layer_network.input_layer_name: input_layer_data.shape}
                single_layer_network._network.reshape(new_shapes)
                del single_layer_network._exec_network
                single_layer_network._exec_network = self._normalizer.plugin.load(network=single_layer_network._network)

            single_layer_network_result = \
                single_layer_network.exec_network.infer({single_layer_network.input_layer_name: input_layer_data})
            difference = compare_nrmsd(single_layer_network_result[single_layer_network.output_layer_name]
                                       , raw_data[single_layer_network.reference_output_layer_name])
            accuracy_drop_list = np.append(accuracy_drop_list, difference)

        return accuracy_drop_list

    def set_accuracy_drop_dict(self):
        if self._accuracy_drop_dict is None:
            self._accuracy_drop_dict = dict()
            for layer in self._network.layers.values():
                if layer.name not in self._ignore_layer_names and \
                        self._normalizer.is_quantization_supported(layer.type):
                    self._accuracy_drop_dict[layer.name] = np.array([])

    def set_single_layer_networks(self):
        assert self._configuration is not None, "Configuration should be set"
        assert self._per_layer_statistics is not None, "Statistics should be set"

        network_info = NetworkInfo(self._configuration.model)

        index = 1
        for layer in self._network.layers.values():
            if layer.name not in self._ignore_layer_names and \
                    self._normalizer.is_quantization_supported(layer.type):
                layer_info = network_info.get_layer(layer.name)
                if (len(layer_info.outputs) == 1) and (len(layer_info.outputs[0].layer.inputs) == 1):
                    activation_layer = self._network.layers[layer_info.outputs[0].layer.name] if \
                        (len(layer_info.outputs) == 1 and
                         self._normalizer.is_quantization_fusing_supported(layer_info,
                                                                           layer_info.outputs[
                                                                               0].layer)) else None
                    if activation_layer:
                        debug("create network #{} for layer {} ({}) -> {} ({})".format(index, layer.name,
                                                                                       layer.type,
                                                                                       activation_layer.name,
                                                                                       activation_layer.type))
                    else:
                        debug("create network #{} for layer {} ({})".format(index, layer.name,
                                                                            layer.type))

                    layer_network, reference_output_layer_name = self._normalizer.create_network_for_layer(
                        None,
                        layer,
                        layer_info,
                        activation_layer)

                    Network.reshape(layer_network, self._configuration.batch_size)

                    network_stats = {}
                    # TODO: initialize only neccessary statistic
                    for layer_name, node_statistic in self._per_layer_statistics.items():
                        network_stats[layer_name] = ie.LayerStats(min=tuple(node_statistic.min_outputs),
                                                                  max=tuple(node_statistic.max_outputs))
                    layer_network.stats.update(network_stats)

                    params = layer_network.layers[layer.name].params
                    params["quantization_level"] = 'I8' if self._configuration.precision == 'INT8' else self._configuration.precision
                    layer_network.layers[layer.name].params = params

                    exec_network = self._normalizer.plugin.load(
                        network=layer_network,
                        config={"EXCLUSIVE_ASYNC_REQUESTS": "YES"})

                    if len(layer_network.inputs) != 1:
                        raise ValueError("created network has several inputs")

                    network_input_layer_name = next(iter(layer_network.inputs.keys()))

                    single_layer_network = SingleLayerNetwork(
                            network=layer_network,
                            exec_network=exec_network,
                            input_layer_name=network_input_layer_name,
                            layer_name=layer.name,
                            output_layer_name=layer.name + "_",
                            reference_output_layer_name=reference_output_layer_name
                    )

                    self._single_layer_networks = np.append(self._single_layer_networks, single_layer_network)
                    self._layers_to_return_to_fp32 = np.append(self._layers_to_return_to_fp32, layer)
                    index += 1

    def callback(self, value, latency=None, **kwargs):

        collect_value = dict()
        for layer_name in value:
            if layer_name in self._collect_layers:
                collect_value[layer_name] = value[layer_name]


        infer_raw_results = []
        infer_raw_results.append(collect_value)

        if not self._callback_preparation:
            self.set_single_layer_networks()
            self.set_accuracy_drop_dict()
            self._callback_preparation = True

        self.calculate_accuracy_drop(infer_raw_results)

    @property
    def aggregated_statistics(self) -> AggregatedStatistics:
        return AggregatedStatistics()

    @property
    def infer_raw_result(self) -> InferRawResults:
        return InferRawResults()

    @property
    def latencies(self) -> list:
        return []

    def release(self):
        pass
