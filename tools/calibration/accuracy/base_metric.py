"""
Copyright (C) 2019 Intel Corporation

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

class BaseMetric:
    def __init__(self, scale, postfix, value):
        self._value = value
        self._symbol = postfix
        self._scale = scale

    @property
    def value(self):
        return self._value

    @property
    def symbol(self):
        raise self._symbol

    def is_achieved(self, reference_accuracy, threshold : float) -> bool:
        raise NotImplementedError()

    def calculate_drop(self, reference_accuracy) -> float:
        raise NotImplementedError()

    def is_better(self, best_accuracy, fp32_accuracy) -> bool:
        raise NotImplementedError()
