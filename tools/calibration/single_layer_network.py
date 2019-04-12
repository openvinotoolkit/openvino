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

from openvino.inference_engine import IENetwork, ExecutableNetwork, InferRequest


# TODO: network and request are not used
# TODO: refactor: create network before inference only
class SingleLayerNetwork:
    '''
    One layer network description
    '''

    def __init__(
        self,
        network: IENetwork,
        exec_network: ExecutableNetwork,
        input_layer_name: str,
        layer_name: str,
        output_layer_name: str,
        reference_output_layer_name: str):

        self._network = network
        self._exec_network = exec_network
        self._input_layer_name = input_layer_name
        self._layer_name = layer_name
        self._output_layer_name = output_layer_name
        self._reference_output_layer_name = reference_output_layer_name
        self._int8_accuracy_list = list()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self._network:
            del self._network
            self._network = None

        if self._exec_network:
            del self._exec_network
            self._exec_network = None

    @property
    def network(self) -> IENetwork:
        return self._network

    @property
    def exec_network(self) -> ExecutableNetwork:
        return self._exec_network

    @property
    def input_layer_name(self) -> str:
        return self._input_layer_name

    @property
    def layer_name(self) -> str:
        return self._layer_name

    @property
    def output_layer_name(self) -> str:
        return self._output_layer_name

    @property
    def reference_output_layer_name(self) -> str:
        return self._reference_output_layer_name

    @property
    def int8_accuracy_list(self) -> list:
        return self._int8_accuracy_list
