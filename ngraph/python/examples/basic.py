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
"""Usage example for the ngraph Pythonic API."""

import numpy as np
import ngraph as ng

A = ng.parameter(shape=[2, 2], name="A", dtype=np.float32)
B = ng.parameter(shape=[2, 2], name="B")
C = ng.parameter(shape=[2, 2], name="C")
# >>> print(A)
# <Parameter: 'A' ([2, 2], float)>

model = (A + B) * C
# >>> print(model)
# <Multiply: 'Multiply_14' ([2, 2])>

runtime = ng.runtime(backend_name="CPU")
# >>> print(runtime)
# <Runtime: Backend='CPU'>

computation = runtime.computation(model, A, B, C)
# >>> print(computation)
# <Computation: Multiply_14(A, B, C)>

value_a = np.array([[1, 2], [3, 4]], dtype=np.float32)
value_b = np.array([[5, 6], [7, 8]], dtype=np.float32)
value_c = np.array([[9, 10], [11, 12]], dtype=np.float32)

result = computation(value_a, value_b, value_c)
# >>> print(result)
# [[ 54.  80.]
#  [110. 144.]]

print("Result = ", result)
