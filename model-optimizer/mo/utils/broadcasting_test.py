"""
 Copyright (C) 2018-2020 Intel Corporation

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
from mo.front.common.partial_infer.utils import int64_array
from mo.utils.broadcasting import bi_directional_broadcasting, bi_directional_shape_broadcasting, uni_directional_broadcasting, uni_directional_shape_broadcasting
from generator import generator, generate
import numpy as np
import unittest


@generator
class TestingBroadcasting(unittest.TestCase):
    @generate(*[([], [20, 30, 10], [20, 30, 10]),
                ([1], [20, 30, 10], [20, 30, 10]),
                ([1, 1, 10], [20, 30, 10], [20, 30, 10]),
                ([20, 1, 10], [20, 30, 10], [20, 30, 10]),
                ([20, 30, 10], [20, 30, 10], [20, 30, 10]),
                ([20, 30, 10], [5, 7, 20, 30, 10], [5, 7, 20, 30, 10]),
                ([1, 2], [20, 3, 10, 2], [20, 3, 10, 2]),
                ([1, 1], [1], None),
                ([5, 10], [1, 10], None),
                ])
    def test_uni_directional_broadcasting(self, input_shape, target_shape, expected_shape):
        self.assertTrue(np.array_equal(uni_directional_shape_broadcasting(input_shape, target_shape), expected_shape))

        input_value = np.array(np.random.rand(*input_shape))
        if expected_shape is not None:
            expected_value = np.broadcast_to(input_value, int64_array(target_shape))
            self.assertTrue(np.array_equal(uni_directional_broadcasting(input_value, int64_array(target_shape)), expected_value))
        else:
            with self.assertRaisesRegex(Exception, '.*cannot be uni-directionally broadcasted.*'):
                uni_directional_broadcasting(input_value, int64_array(target_shape))
