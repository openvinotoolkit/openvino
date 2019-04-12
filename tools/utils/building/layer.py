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
from ..biases import Biases
from ..weights import Weights


class Layer:
    TEMPLATE = (
        '<layer name="{name}" type="{type}" precision="FP32" id="{id}">'
            '{data}'
            '{input}'
            '{output}'
            '{weights}'
            '{biases}'
        '</layer>')

    def __init__(
        self, id: int,
        type: str,
        name: str,
        params: dict,
        input_dims: list,
        output_dims: list,
        weights: Weights = None,
        biases: Biases = None):
        self._id = id
        self._type = type
        self._name = name
        self._params = params
        self._input_dims = input_dims
        self._output_dims = output_dims
        self._weights = weights
        self._biases = biases

    @property
    def id(self) -> str:
        return self._id

    @property
    def type(self) -> str:
        return self._type

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return self._params

    @property
    def input_dims(self) -> list:
        return self._input_dims

    @property
    def output_dims(self) -> list:
        return self._output_dims

    @property
    def weights(self) -> Weights:
        return self._weights

    @property
    def biases(self) -> Biases:
        return self._biases

    def _output_dims_to_xml(self) -> str:
        if self._output_dims:
            if len(self._output_dims) == 2:
                output_xml = (
                    '<output>'
                        '<port id="1">'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                        '</port>'
                    '</output>').format(self._output_dims[0], self._output_dims[1])
            elif len(self._output_dims) == 4:
                output_xml = (
                    '<output>'
                        '<port id="1">'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                        '</port>'
                    '</output>').format(self._output_dims[0], self._output_dims[1], self._output_dims[2], self._output_dims[3])
            else:
                raise NotImplementedError("{} dimensions for outputs (layer name '{}', type '{}') are not supported".format(
                    len(self._output_dims),
                    self._name,
                    self._type))
        else:
            output_xml = None
        return output_xml

    def _input_dims_to_xml(self) -> str:
        if self._input_dims:
            if len(self._input_dims) == 2:
                input_xml = (
                    '<input>'
                        '<port id="0">'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                        '</port>'
                    '</input>').format(self._input_dims[0], self._input_dims[1])
            elif len(self._input_dims) == 4:
                input_xml = (
                    '<input>'
                        '<port id="0">'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                            '<dim>{}</dim>'
                        '</port>'
                    '</input>').format(self._input_dims[0], self._input_dims[1], self._input_dims[2], self._input_dims[3])
            else:
                raise NotImplementedError("{} dimensions for inputs (layer name '{}', type '{}') are not supported".format(
                    len(self._input_dims),
                    self._name,
                    self._type))
        else:
            input_xml = None

        return input_xml

    def __str__(self) -> str:
        if self._params:
            data_xml = "<data "
            for param_key in self._params.keys():
                data_xml += '{}="{}" '.format(param_key, self._params[param_key])
            data_xml += " />"
        else:
            data_xml = None

        return self.TEMPLATE.format(
            name=self._name,
            type=self._type,
            id=self._id,
            data=(data_xml if data_xml else ''),
            input=(self._input_dims_to_xml() if self._input_dims else ''),
            output=(self._output_dims_to_xml() if self._output_dims else ''),
            weights=('<weights offset="{offset}" size="{size}"/>'.format(offset=self._weights.offset, size=self._weights.size) if self._weights else ''),
            biases=('<biases offset="{offset}" size="{size}"/>'.format(offset=self._biases.offset, size=self._biases.size) if self._biases else '')
            )
