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

import os
import tempfile
import shutil
import ntpath

import openvino.inference_engine as ie
from .utils.path import Path


class Network:
    @staticmethod
    def reload(model_path: str, statistics = None, quantization_levels: dict() = None, batch_size: int = None):
        tmp_model_dir = None
        try:
            with Network(model_path) as network:
                if statistics:
                    network.set_statistics(statistics)
                if quantization_levels:
                    network.set_quantization_levels(quantization_levels)

                tmp_model_dir = tempfile.mkdtemp(".model")
                tmp_model_path = os.path.join(tmp_model_dir, ntpath.basename(model_path))
                network.serialize(tmp_model_path)

            network = Network(tmp_model_path)
            Network.reshape(network.ie_network, batch_size)
            return network
        finally:
            if tmp_model_dir:
                shutil.rmtree(tmp_model_dir)

    def __init__(self, model_path: str, weights_path: str=None):
        if model_path is None:
            raise ValueError("model_path is None")

        self._model_path = model_path
        self._weights_path = weights_path if weights_path else Path.get_weights(model_path)
        self._ie_network = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def release(self):
        if self._ie_network:
            del self._ie_network
            self._ie_network = None

    @staticmethod
    def reshape(ie_network: ie.IENetwork, batch_size: int) -> ie.IENetwork:
        if batch_size and batch_size != ie_network.batch_size:
            new_shapes = {}
            for input_layer_name, input_layer in ie_network.inputs.items():
                layout = input_layer.layout
                if layout == 'C':
                    new_shape = (input_layer.shape[0],)
                elif layout == 'NC':
                    new_shape = (batch_size, input_layer.shape[1])
                else:
                    raise ValueError("not supported layout '{}'".format(layout))                    
                new_shapes[input_layer_name] = new_shape
            ie_network.reshape(new_shapes)
        return ie_network

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def weights_path(self) -> str:
        return self._weights_path

    @property
    def ie_network(self) -> ie.IENetwork:
        if not self._ie_network:
            self._ie_network = ie.IENetwork(self._model_path, self._weights_path)
        return self._ie_network

    def set_quantization_levels(self, quantization_level: dict):
        for layer_name, value in quantization_level.items():
            params = self.ie_network.layers[layer_name].params
            params["quantization_level"] = value
            self.ie_network.layers[layer_name].params = params

    def set_statistics(self, statistics: dict):
        network_stats = {}
        for layer_name, node_statistic in statistics.items():
            network_stats[layer_name] = ie.LayerStats(min=tuple(node_statistic.min_outputs),
                                                      max=tuple(node_statistic.max_outputs))
        self.ie_network.stats.update(network_stats)

    def serialize(self, model_path: str, weights_path: str=None):
        self.ie_network.serialize(model_path, weights_path if weights_path else Path.get_weights(model_path))
