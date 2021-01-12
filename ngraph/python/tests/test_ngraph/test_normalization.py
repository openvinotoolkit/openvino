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
from tests import xfail_issue_40957


@xfail_issue_40957
def test_lrn():
    input_image_shape = (2, 3, 2, 1)
    input_image = np.arange(int(np.prod(input_image_shape))).reshape(input_image_shape).astype("f")
    axes = np.array([1], dtype=np.int64)
    runtime = get_runtime()
    model = ng.lrn(ng.constant(input_image), ng.constant(axes), alpha=1.0, beta=2.0, bias=1.0, size=3)
    computation = runtime.computation(model)
    result = computation()
    assert np.allclose(
        result,
        np.array(
            [
                [[[0.0], [0.05325444]], [[0.03402646], [0.01869806]], [[0.06805293], [0.03287071]]],
                [[[0.00509002], [0.00356153]], [[0.00174719], [0.0012555]], [[0.00322708], [0.00235574]]],
            ],
            dtype=np.float32,
        ),
    )

    # Test LRN default parameter values
    model = ng.lrn(ng.constant(input_image), ng.constant(axes))
    computation = runtime.computation(model)
    result = computation()
    assert np.allclose(
        result,
        np.array(
            [
                [[[0.0], [0.35355338]], [[0.8944272], [1.0606602]], [[1.7888544], [1.767767]]],
                [[[0.93704253], [0.97827977]], [[1.2493901], [1.2577883]], [[1.5617375], [1.5372968]]],
            ],
            dtype=np.float32,
        ),
    )


def test_lrn_factory():
    alpha = 0.0002
    beta = 0.5
    bias = 2.0
    nsize = 3
    axis = np.array([1], dtype=np.int32)
    x = np.array(
        [
            [
                [
                    [0.31403765, -0.16793324, 1.388258, -0.6902954],
                    [-0.3994045, -0.7833511, -0.30992958, 0.3557573],
                    [-0.4682631, 1.1741459, -2.414789, -0.42783254],
                ],
                [
                    [-0.82199496, -0.03900861, -0.43670088, -0.53810567],
                    [-0.10769883, 0.75242394, -0.2507971, 1.0447186],
                    [-1.4777364, 0.19993274, 0.925649, -2.282516],
                ],
            ]
        ],
        dtype=np.float32,
    )
    excepted = np.array(
        [
            [
                [
                    [0.22205527, -0.11874668, 0.98161197, -0.4881063],
                    [-0.2824208, -0.553902, -0.21915273, 0.2515533],
                    [-0.33109877, 0.8302269, -1.7073234, -0.3024961],
                ],
                [
                    [-0.5812307, -0.02758324, -0.30878326, -0.38049328],
                    [-0.07615435, 0.53203356, -0.17733987, 0.7387126],
                    [-1.0448756, 0.14137045, 0.6544598, -1.6138376],
                ],
            ]
        ],
        dtype=np.float32,
    )
    result = run_op_node([x], ng.lrn, axis, alpha, beta, bias, nsize)

    assert np.allclose(result, excepted)


def test_batch_norm_inference():
    data = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float32)
    gamma = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    beta = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    variance = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    epsilon = 9.99e-06
    excepted = np.array([[2.0, 6.0, 12.0], [-2.0, -6.0, -12.0]], dtype=np.float32)

    result = run_op_node([data, gamma, beta, mean, variance], ng.batch_norm_inference, epsilon)

    assert np.allclose(result, excepted)


def test_mvn():
    data = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
                      [[0.02916367], [0.12964272], [0.5060197]],
                      [[0.79538304], [0.9411346], [0.9546573]]],
                     [[[0.17730942], [0.46192095], [0.26480448]],
                      [[0.6746842], [0.01665257], [0.62473077]],
                      [[0.9240844], [0.9722341], [0.11965699]]],
                     [[[0.41356155], [0.9129373], [0.59330076]],
                      [[0.81929934], [0.7862604], [0.11799799]],
                      [[0.69248444], [0.54119414], [0.07513223]]]], dtype=np.float32)
    axes = np.array([2, 3], dtype=np.int64)
    epsilon = 1e-9
    normalize_variance = True
    eps_mode = 'outside_sqrt'
    excepted = np.array([[[[1.3546423], [0.33053496], [-1.5450814]],
                          [[-1.2106764], [-0.8925952], [0.29888135]],
                          [[0.38083088], [0.81808794], [0.85865635]]],
                         [[[-1.1060555], [-0.05552877], [-0.78310335]],
                          [[0.83281356], [-1.250282], [0.67467856]],
                          [[0.7669372], [0.9113869], [-1.6463585]]],
                         [[[-0.23402764], [1.6092131], [0.42940593]],
                          [[1.2906139], [1.1860244], [-0.92945826]],
                          [[0.0721334], [-0.38174], [-1.7799333]]]], dtype=np.float32)

    result = run_op_node([data, axes], ng.mvn, normalize_variance, epsilon, eps_mode)

    assert np.allclose(result, excepted)
