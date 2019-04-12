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

import collections

from .biases import Biases
from .weights import Weights
from .port import Port


class Layer:
    def __init__(self, data: dict):
        self._id = int(data['id'])
        self._name = data['name']
        self._precision = data['precision']
        self._type = data['type']

        self._input_ports = Layer.__init_ports(data, 'input')
        self._output_ports = Layer.__init_ports(data, 'output')

        self._inputs = list()
        self._outputs = list()

        blobs = data['blobs'] if 'blobs' in data else data
        self._weights = Weights(int(blobs['weights']['offset']), int(blobs['weights']['size'])) if 'weights' in blobs else Weights(0, 0)
        self._biases = Biases(int(blobs['biases']['offset']), int(blobs['biases']['size'])) if 'biases' in blobs else Biases(0, 0)

    @staticmethod
    def __init_ports(data: dict, key: str) -> dict:
        result_ports = dict()
        if (key in data) and ('port' in data[key]):            
            ports = data[key]['port']
            if type(ports) is list:
                for port_dict in ports:
                    id = int(port_dict['id'])
                    result_ports[id] = Port(id, list(map(int, port_dict['dim'])))
            elif type(ports) is collections.OrderedDict:
                id = int(ports['id'])
                result_ports[id] = Port(id, list(map(int, ports['dim'])))
            else:
                raise ValueError("unexpected ports type '{}'".format(type(ports)))
        return result_ports

    def init(self, inputs: list, outputs: list):
        self._inputs = inputs
        self._outputs = outputs

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def precision(self) -> str:
        return self._precision

    @property
    def type(self) -> str:
        return self._type

    @property
    def input_ports(self):
        return self._input_ports

    @property
    def output_ports(self):
        return self._output_ports

    @property
    def inputs(self) -> list:
        return self._inputs

    @property
    def outputs(self) -> list:
        return self._outputs

    @property
    def weights(self):
        return self._weights

    @property
    def biases(self):
        return self._biases
