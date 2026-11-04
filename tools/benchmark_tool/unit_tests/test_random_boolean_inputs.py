# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
from openvino import PartialShape, Type
from openvino.tools.benchmark.utils.inputs_filling import fill_tensors_with_random
from openvino.tools.benchmark.utils.utils import AppInputInfo


class TestRandomBooleanInputs(unittest.TestCase):
    def input_info(self, shape, element_type=Type.boolean):
        info = AppInputInfo()
        info.element_type = element_type
        info.partial_shape = PartialShape(shape)
        return info

    def test_boolean_tensors_contain_both_values(self):
        for shape in [[128], [8, 16]]:
            with self.subTest(shape=shape):
                tensor = fill_tensors_with_random(self.input_info(shape))[0]
                self.assertEqual(tensor.element_type, Type.boolean)
                self.assertEqual(list(tensor.shape), shape)
                np.testing.assert_array_equal(np.unique(tensor.data), [False, True])

    def test_boolean_scalar_keeps_its_rank_and_type(self):
        tensor = fill_tensors_with_random(self.input_info([]))[0]
        self.assertEqual(tensor.element_type, Type.boolean)
        self.assertEqual(list(tensor.shape), [])

    def test_boolean_generation_is_repeatable(self):
        info = self.input_info([128])
        first = fill_tensors_with_random(info)[0]
        second = fill_tensors_with_random(info)[0]
        np.testing.assert_array_equal(first.data, second.data)

    def test_float_inputs_keep_their_type_and_fractional_values(self):
        shape = [8, 16]
        tensor = fill_tensors_with_random(self.input_info(shape, Type.f32))[0]
        self.assertEqual(tensor.element_type, Type.f32)
        self.assertEqual(list(tensor.shape), shape)
        self.assertTrue(np.any(tensor.data != np.floor(tensor.data)))
