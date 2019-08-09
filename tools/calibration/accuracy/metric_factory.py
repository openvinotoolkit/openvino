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

from .regression_metrics import RegressionMetrics
from .metric_in_percent import MetricInPercent
from .base_metric import BaseMetric

class MetricFactory:
    @staticmethod
    def create(scale : int, postfix : str, accuracy_val : float) -> BaseMetric:
        if postfix == '%':
            return MetricInPercent(scale, postfix, accuracy_val)
        return RegressionMetrics(scale, postfix, accuracy_val)
