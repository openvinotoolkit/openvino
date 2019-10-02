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

from .base_calibrator import BaseCalibrator
from .calibrator_configuration import CalibratorConfiguration


# TODO: not comlpeted. Some methods will be moved from Calibrator and customized to INT8
class Int8Calibrator(BaseCalibrator):
    '''
    INT8 calibrator
    '''
    def __init__(self, configuration: CalibratorConfiguration):
        super().__init__(configuration)

    @property
    def precision(self):
        return "INT8"

    def is_quantization_supported(self, layer_type: str) -> bool:
        return layer_type.lower() == "convolution" or layer_type.lower() == "fullyconnected"
