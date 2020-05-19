#!/usr/bin/env python
# ******************************************************************************
# Copyright 2017-2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import numpy as np

input = np.arange(1, 25, 1).reshape(1, 2, 3, 4).astype(np.float32)
eps = np.array([1e-6]).astype(np.float32)
# across chw axes
norm = np.sqrt(np.sum(np.power(input, 2), axis=(1), keepdims=True) + eps)
result = input/norm

for elem in np.nditer(result):
    print(str(round(elem, 8)) + 'f, ')
