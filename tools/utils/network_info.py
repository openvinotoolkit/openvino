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

import xmltodict
from typing import List

from .layer import Layer
from .edge import Edge
from .connection import Connection


# TODO: custom implementation:
# 1. get in/out layers
# 2. add_layer
class NetworkInfo:
    def __init__(self, model_path: str):
        
        model_content = None
        with open(model_path, 'r') as mode_file:
            model_content = mode_file.read()

        model_xml = xmltodict.parse(model_content, attr_prefix='')
        if 'net' not in model_xml:
            raise ValueError("IR file '{}' format is not correct".format(model_path))

        self._model = model_xml['net']

        # TODO: move to private method
        ordered_edges = self._model['edges']['edge']
        self._edges_by_from_layer = dict()
        self._edges_by_to_layer = dict()
        for ordered_edge in ordered_edges:
            from_layer = int(ordered_edge['from-layer'])
            to_layer = int(ordered_edge['to-layer'])

            edge = Edge(ordered_edge)

            if from_layer not in self._edges_by_from_layer:
                self._edges_by_from_layer[from_layer] = list()
            self._edges_by_from_layer[from_layer].append(edge)

            if to_layer not in self._edges_by_to_layer:
                self._edges_by_to_layer[to_layer] = list()
            self._edges_by_to_layer[to_layer].append(edge)

        # TODO: move to private method
        ordered_layers = self._model['layers']['layer']
        self._layer_by_id = dict()
        self._layer_by_name = dict()
        for ordered_layer in ordered_layers:
            layer = Layer(ordered_layer)
            self._layer_by_id[int(ordered_layer['id'])] = layer
            self._layer_by_name[layer.name] = layer

        # TODO: move to private method
        for layer_id, layer in self._layer_by_id.items():
            input_edges = self._edges_by_to_layer[layer_id] if layer_id in self._edges_by_to_layer else list()
            inputs = list()
            for edge in input_edges:
                if edge.from_layer not in self._layer_by_id:
                    raise ValueError("layer with id {} was not found".format(edge.from_layer))

                # inputs.append(self._layer_by_id[edge.from_layer])
                from_layer = self._layer_by_id[edge.from_layer]
                inputs.append(Connection(edge=edge, port=layer.input_ports[edge.to_port], layer=from_layer))

            output_edges = self._edges_by_from_layer[layer_id] if layer_id in self._edges_by_from_layer else list()
            outputs = list()
            for edge in output_edges:
                if edge.to_layer not in self._layer_by_id:
                    raise ValueError("layer with id {} was not found".format(edge.to_layer))

                # outputs.append(self._layer_by_id[edge.to_layer])
                to_layer = self._layer_by_id[edge.to_layer]
                outputs.append(Connection(edge=edge, port=layer.output_ports[edge.from_port], layer=to_layer))

            layer.init(inputs, outputs)

        pass

    def get_layer_names_by_types(self, layer_types: List[str]) -> List[str]:
        layer_names = []
        if layer_types:
            for layer in self._layer_by_name.values():
                if layer.type in layer_types:
                    layer_names.append(layer.name)
        return layer_names

    @property
    def layers(self) -> int:
        return self._layer_by_id

    def get_layer(self, layer_name: str) -> Layer:
        return self._layer_by_name[layer_name]

    def explore_inputs(self, layer: Layer, expected_input_types: List[str]) -> bool:
        for layer_input in layer.inputs:
            if layer_input.layer.type not in expected_input_types:
                return False
            if not self.explore_inputs(layer_input.layer, expected_input_types):
                return False
        return True

    @property
    def inputs(self):
        inputs = dict()
        for id, layer in self.layers.items():
            if layer.type == 'Input':
                inputs[id] = layer
        return inputs
