# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.runtime.opset8 as ov
import numpy as np
from tests.runtime import get_runtime


def test_random_uniform():
    runtime = get_runtime()
    input_tensor = ov.constant(np.array([2, 4, 3], dtype=np.int32))
    min_val = ov.constant(np.array([-2.7], dtype=np.float32))
    max_val = ov.constant(np.array([3.5], dtype=np.float32))

    random_uniform_node = ov.random_uniform(input_tensor, min_val, max_val,
                                            output_type="f32", global_seed=7461,
                                            op_seed=1546)
    computation = runtime.computation(random_uniform_node)
    random_uniform_results = computation()
    expected_results = np.array([[[2.8450181, -2.3457108, 2.2134445],
                                  [-1.0436587, 0.79548645, 1.3023183],
                                  [0.34447956, -2.0267959, 1.3989122],
                                  [0.9607613, 1.5363653, 3.117298]],

                                 [[1.570041, 2.2782724, 2.3193843],
                                  [3.3393657, 0.63299894, 0.41231918],
                                  [3.1739233, 0.03919673, -0.2136085],
                                  [-1.4519991, -2.277353, 2.630727]]], dtype=np.float32)

    assert np.allclose(random_uniform_results, expected_results)
