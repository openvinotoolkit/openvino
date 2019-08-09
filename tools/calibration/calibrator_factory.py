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

from .calibrator_configuration import CalibratorConfiguration
from .int8_calibrator import Int8Calibrator


class CalibratorFactory:
    @staticmethod
    def create(precision: str, configuration: CalibratorConfiguration):
        if precision.lower() == "int8":
            return Int8Calibrator(configuration)

        raise ValueError("not supported precision '{}'".format(precision))
