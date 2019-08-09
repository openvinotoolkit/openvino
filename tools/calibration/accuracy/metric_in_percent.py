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
from .base_metric import BaseMetric

class MetricInPercent(BaseMetric):
    def __init__(self, scale, postfix, value):
        super().__init__(scale, postfix, value)

    @property
    def value(self) -> float:
        return self._value * self._scale

    @property
    def symbol(self) -> str:
        return self._symbol

    def is_achieved(self, reference_accuracy : BaseMetric, threshold : float) -> bool:
        return (reference_accuracy.value - self.value) < threshold

    def calculate_drop(self, reference_accuracy : BaseMetric) -> float:
        return reference_accuracy.value - self.value

    def is_better(self, best_accuracy : BaseMetric, fp32_accuracy : BaseMetric) -> bool:
        if best_accuracy is None or self.value > best_accuracy.value:
            return True
        return False
