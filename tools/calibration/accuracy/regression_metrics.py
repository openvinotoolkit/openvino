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

class RegressionMetrics(BaseMetric):
    def __init__(self, scale, postfix, value):
        super().__init__(scale, postfix, value)

    @property
    def value(self) -> float:
        return self._value * self._scale

    @property
    def symbol(self) -> str:
        return self._symbol

    def is_achieved(self, reference_accuracy : BaseMetric, threshold : float) -> bool:
        is_achieved_value = abs(reference_accuracy.value - self.value) * 100 / reference_accuracy.value < threshold
        return is_achieved_value

    def calculate_drop(self, reference_accuracy : BaseMetric) -> float:
        return abs(reference_accuracy.value - self.value) * 100 / reference_accuracy.value

    def is_better(self, best_accuracy : BaseMetric, fp32_accuracy : BaseMetric) -> bool:
        if best_accuracy is None:
            return True
        return self.calculate_drop(fp32_accuracy) < best_accuracy.calculate_drop(fp32_accuracy)