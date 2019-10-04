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

import json
import numpy
import openvino.inference_engine as ie
from .network_node_stats import NetworkNodeStats
from .shape import Shape


class AggregatedStatistics:
    INDEX_MIN = 0
    INDEX_MAX = 1

    def __init__(self, result=None, ignore_layer_names: set=None, iterations_count: int = 1, dataset_size: int = 1):
        self._ignore_layer_names = ignore_layer_names
        self._registered_layers = None
        self._iterations_count = iterations_count
        self._dataset_size = dataset_size
        self._itteration = 0

        if result:
            for inference_result in result.result:
                self.add(network = result.network, exec_network = result.exec_network, inference_result = inference_result)

    def release(self):
        if self._registered_layers:
            del self._registered_layers
            self._registered_layers = None

    def add(
        self,
        network: ie.IENetwork,
        exec_network: ie.ExecutableNetwork,
        inference_result
    ):
        '''
        Add inference result to aggregated statistics instance
        '''
        layer_names = network.layers.keys()

        if not self._registered_layers:
            self._registered_layers = dict()
            initialized = False
        else:
            initialized = True

        # TODO: can be refactored: we are itterating by all layers (to cover input layers output) to collect statistics
        # for inference_result in inference_results:
        for out_layer_name in layer_names:
            if self._ignore_layer_names and out_layer_name in self._ignore_layer_names or \
                    out_layer_name in network.outputs and network.outputs[out_layer_name].layout.lower() == 'blocked':
                continue

            if out_layer_name in network.inputs:
                output_blob = exec_network.requests[0].inputs[out_layer_name]
                shape = Shape.create(network.inputs[out_layer_name].layout, output_blob.shape)
            else:
                # TODO: can be refactored: we are itterating by all layers (to cover input layers output) to collect statistics
                if out_layer_name not in inference_result:
                    continue
                output_blob = inference_result[out_layer_name]
                shape = Shape.create(network.outputs[out_layer_name].layout, output_blob.shape)

            if not initialized:
                # for const layers N is not equal batch size
                # self._registered_layers[out_layer_name] = numpy.empty((shape.c, self._dataset_size, 2))
                self._registered_layers[out_layer_name] = numpy.empty((shape.c, shape.n * self._iterations_count, 2))

            if shape.layout[0] != 'C' and not (len(shape.layout) >= 2 and shape.layout[0] == 'N' and shape.layout[1] == 'C'):
                raise ValueError("unsupported layout '{}'".format(shape.layout))

            if shape.layout[0] != 'N':
                output_blob = [output_blob]

            for sample in range(0, shape.n):
                for channel in range(0, shape.c):
                    self.add_tensor_statistics(out_layer_name, output_blob, shape.n, sample, channel, self._itteration)

        self._itteration += 1

    def register_layer(self, layer_name: str):
        if layer_name in self._registered_layers:
            raise ValueError("layer '{}' has been added already".format(layer_name))

        self._registered_layers[layer_name] = None

    @property
    def registered_layers(self):
        return self._registered_layers

    def add_tensor_statistics(self, layer_name: str, data, n: int, sample: int, channel: int, itteration: int):
        channels = self._registered_layers[layer_name]

        n_index = sample + n * itteration
        if n_index >= channels.shape[1]:
            channels.resize((channels.shape[0], n_index + 1, channels.shape[2]), refcheck=False)
        if channel >= channels.shape[0]:
            channels.resize((channel + 1, channels.shape[1], channels.shape[2]), refcheck=False)

        channels.itemset((channel, n_index, self.INDEX_MIN), data[sample][channel].min())
        channels.itemset((channel, n_index, self.INDEX_MAX), data[sample][channel].max())

    def get_number_channels(self, layer_name: str):
        if layer_name in self._registered_layers:
            return len(self._registered_layers[layer_name])
        return 0

    def get_data_min_max(self, layer_name: str, channel: int, threshold: float = None):
        # take data by name
        if layer_name in self._registered_layers:
            layer = self._registered_layers[layer_name]
            stats = layer[channel]

            # having absolute min/max values, we can create new statistic
            max_values = list()
            min_values = list()
            for tensor_statistic in stats:
                max_values.append(tensor_statistic.item(self.INDEX_MAX))
                min_values.append(tensor_statistic.item(self.INDEX_MIN))

            # define number of elements to throw out
            element_to_take = int(len(max_values) * threshold / 100) if threshold else len(max_values)
            elements_to_throw = len(max_values) - element_to_take if threshold else 0

            element_to_take = len(max_values) if element_to_take > len(max_values) else element_to_take
            elements_to_throw = (len(min_values) - 1) if elements_to_throw >= len(min_values) else elements_to_throw

            max_values.sort()
            min_values.sort()

            min = min_values[elements_to_throw]
            max = max_values[element_to_take - 1]
        else:
            min = max = 0.0

        return min, max

    def serialize(self, json_file_path: str):
        with open(json_file_path, 'w') as out_file:
            json.dump(self._registered_layers, out_file)


    def get_node_statistics(self, threshold = None):
        net_nodes_stats = dict()
        # go over all outputs and get aggregated statistics
        for layer_name in self.registered_layers:
            channels_count = self.get_number_channels(layer_name)

            if layer_name not in net_nodes_stats:
                node_stats = NetworkNodeStats(channels_count)
                net_nodes_stats[layer_name] = node_stats
            else:
                node_stats = net_nodes_stats[layer_name]

            for channel in range(channels_count):
                node_stats.min_outputs[channel], node_stats.max_outputs[channel] = self.get_data_min_max(layer_name, channel, threshold)

        return net_nodes_stats

    def pop(self, ignore_layer_names: set):
        for ignore_layer_name in ignore_layer_names:
            self._registered_layers.pop(ignore_layer_name)