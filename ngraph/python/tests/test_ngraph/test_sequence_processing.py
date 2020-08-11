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

import ngraph as ng
from tests.runtime import get_runtime
from tests.test_ngraph.util import run_op_node
from tests import xfail_issue_34323


def test_onehot():
    runtime = get_runtime()
    param = ng.parameter([3], dtype=np.int32)
    model = ng.one_hot(param, 3, 1, 0, 0)
    computation = runtime.computation(model, param)

    expected = np.eye(3)[np.array([1, 0, 2])]
    input_data = np.array([1, 0, 2], dtype=np.int32)
    result = computation(input_data)
    assert np.allclose(result, expected)


@xfail_issue_34323
def test_one_hot():
    data = np.array([0, 1, 2], dtype=np.int32)
    depth = 2
    on_value = 5
    off_value = 10
    axis = -1
    excepted = [[5, 10], [10, 5], [10, 10]]

    result = run_op_node([data, depth, on_value, off_value], ng.one_hot, axis)
    assert np.allclose(result, excepted)


@xfail_issue_34323
def test_range():
    start = 5
    stop = 35
    step = 5

    result = run_op_node([start, stop, step], ng.range)
    assert np.allclose(result, [5, 10, 15, 20, 25, 30])
