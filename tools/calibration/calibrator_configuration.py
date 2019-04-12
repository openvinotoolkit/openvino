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


class CalibratorConfiguration:
    def __init__(self, configuration):
        self._config = configuration.config
        self._model = configuration.model
        self._weights = configuration.weights
        self._device = configuration.device
        self._cpu_extension = configuration.cpu_extension
        self._gpu_extension = configuration.gpu_extension
        self._threshold = configuration.threshold
        self._batch_size = configuration.batch_size
        self._progress = configuration.progress

    @property
    def config(self) -> str:
        return self._config

    @property
    def model(self) -> str:
        return self._model

    @property
    def weights(self) -> str:
        return self._weights

    @property
    def device(self) -> str:
        return self._device

    @property
    def cpu_extension(self) -> str:
        return self._cpu_extension

    @property
    def gpu_extension(self) -> str:
        return self._gpu_extension

    @property
    def threshold(self) -> str:
        return self._threshold

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def progress(self) -> str:
        return self._progress
