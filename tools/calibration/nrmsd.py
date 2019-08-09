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

import numpy as np


def compare_nrmsd(actual_data, expected_data):
    if actual_data.size != expected_data.size:
        raise ValueError(
            "actual data size {} is not equal expected data size {}".format(actual_data.size, expected_data.size))
    if actual_data.size == 0:
        raise ValueError("actual data size should be positive")
    
    if expected_data.max() - expected_data.min() == 0:
        return 1.0
    sum = np.sum(np.power(expected_data - actual_data, 2))
    sum = sum / expected_data.size
    sum = np.sqrt(sum)
    sum = sum / (expected_data.max() - expected_data.min())
    return sum