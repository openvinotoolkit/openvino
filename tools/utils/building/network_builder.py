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


# TODO: limitations:
# - one input
# - one output
# - dims size is 4
class NetworkBuilder:

    EDGES_TEMPLATE2 = (
        '<edges>'
            '<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>'
            '<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>'
        '</edges>')

    EDGES_TEMPLATE3 = (
        '<edges>'
            '<edge from-layer="0" from-port="1" to-layer="1" to-port="0"/>'
            '<edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>'
            '<edge from-layer="2" from-port="1" to-layer="3" to-port="0"/>'
        '</edges>')

    def __init__(self, version: int = 3):
        self._layers = list()

    def __str__(self):
        # xml = '<net name="one_layer_calibtation_network" version="2" batch="1"><layers>'
        xml = '<net name="one_layer_calibtation_network" version="3" batch="1"><layers>'
        for layer in self._layers:
            xml = xml + str(layer)

        xml = xml + "</layers>" + (NetworkBuilder.EDGES_TEMPLATE2 if len(self._layers) == 3 else NetworkBuilder.EDGES_TEMPLATE3) + "</net>"
        return xml

    def sequential(self, layers):
        self._layers = layers
        return self
