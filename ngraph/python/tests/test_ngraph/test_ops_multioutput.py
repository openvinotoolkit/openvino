# ******************************************************************************
# Copyright 2017-2021 Intel Corporation
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

import ngraph as ng
from tests.runtime import get_runtime


def test_split():
    runtime = get_runtime()
    input_tensor = ng.constant(np.array([0, 1, 2, 3, 4, 5], dtype=np.int32))
    axis = ng.constant(0, dtype=np.int64)
    splits = 3

    split_node = ng.split(input_tensor, axis, splits)
    computation = runtime.computation(split_node)
    split_results = computation()
    expected_results = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    assert np.allclose(split_results, expected_results)


def test_variadic_split():
    runtime = get_runtime()
    input_tensor = ng.constant(np.array([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=np.int32))
    axis = ng.constant(1, dtype=np.int64)
    splits = ng.constant(np.array([2, 4], dtype=np.int64))

    v_split_node = ng.variadic_split(input_tensor, axis, splits)
    computation = runtime.computation(v_split_node)
    results = computation()
    split0 = np.array([[0, 1], [6, 7]], dtype=np.int32)
    split1 = np.array([[2, 3, 4, 5], [8, 9, 10, 11]], dtype=np.int32)

    assert np.allclose(results[0], split0)
    assert np.allclose(results[1], split1)
