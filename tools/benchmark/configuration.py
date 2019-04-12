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


class Configuration:
    def __init__(
        self,
        config: str,
        model: str,
        weights: str,
        device: str,
        cpu_extension: str,
        gpu_extension: str,
        benchmark_iterations_count: int
    ):

        self._config = config
        self._model = model
        self._weights = weights
        self._device = device
        self._cpu_extension = cpu_extension
        self._gpu_extension = gpu_extension
        self._benchmark_iterations_count = benchmark_iterations_count

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
    def benchmark_iterations_count(self):
        return self._benchmark_iterations_count