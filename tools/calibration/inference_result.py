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

from .aggregated_statistics import AggregatedStatistics
from .calibration_metrics import CalibrationMetrics
from .infer_raw_results import InferRawResults


class InferenceResult:
    def __init__(self,
                 result: InferRawResults,
                 metrics: CalibrationMetrics,
                 aggregated_statistics: AggregatedStatistics,
                 performance_counters: dict,
                 layers_accuracy_drop = None):
        self._result = result
        self._metrics = metrics
        self._aggregated_statistics = aggregated_statistics
        self._performance_counters = performance_counters
        self._layers_accuracy_drop = layers_accuracy_drop

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.release()

    def release(self):
        if self._result:
            self._result.release()
            self._result = None

    @property
    def result(self) -> InferRawResults:
        return self._result

    @property
    def metrics(self) -> CalibrationMetrics:
        return self._metrics

    @property
    def aggregated_statistics(self) -> AggregatedStatistics:
        return self._aggregated_statistics

    @property
    def performance_counters(self) -> dict:
        return self._performance_counters

    @property
    def layers_accuracy_drop(self) -> dict:
        return self._layers_accuracy_drop

    def get_class_ids(self, output_layer_name: str) -> list:
        '''
        Return class identifier list for classification networks
        '''

        result_classes_id_list = list()
        for layers_result in self._result:
            if output_layer_name not in layers_result:
                raise KeyError("layer '{}' is not included int results".format(output_layer_name))

            layer_result = layers_result[output_layer_name]
            if layer_result.size == 0:
                raise ValueError("result array is empty")

            max_value = layer_result.item(0)
            max_class_id = 0

            for class_id in range(layer_result.size):
                value = layer_result.item(class_id)
                if value > max_value:
                    max_value = value
                    max_class_id = class_id

            result_classes_id_list.append(max_class_id)

        return result_classes_id_list
