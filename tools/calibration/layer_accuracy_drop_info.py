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


class LayerAccuracyDropInfo:
    def __init__(self, layer_name: str, value: float):
        self._layer_name = layer_name
        self._value = value

    @property
    def layer_name(self):
        return self._layer_name

    @property
    def value(self):
        return self._value

    @staticmethod
    def calculate(accuracy_drop: list) -> float:
        sum = 0.0
        for d in accuracy_drop:
            sum += d
        return sum / len(accuracy_drop)
