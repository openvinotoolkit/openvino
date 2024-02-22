# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import pytest

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, mo_array, is_fully_defined, \
    dynamic_dimension_value, dynamic_dimension, shape_array, compatible_shapes, shape_delete, shape_insert, \
    strict_compare_tensors, clarify_partial_shape
from openvino.tools.mo.utils.error import Error


def gen_masked_array(array, masked_indices):
    """
    Creates a masked array from the input array by masking specific elements.

    :param array: the input array
    :param masked_indices: element indices to be masked
    :return: the result masked array
    """
    res = np.ma.masked_array(array)
    for index in masked_indices:
        res[index] = np.ma.masked
    return res


class TestIsFullyDefinedTest():
    @pytest.mark.parametrize("data, result",[(None, False),
                (int64_array([2, 3, 5, 7]), True),  # int64 array with valid values
                (np.array([2, 3, 5, 7]), True),  # any numpy array with valid values
                (np.array([2, dynamic_dimension_value]), True),  # array with dynamic dimension value is fully defined!
                (shape_array([2, dynamic_dimension_value, 5]), False),  # masked array with at least one masked element
                (shape_array([2, 4, 5]), True),  # masked array with no masked elements is fully defined
                (dynamic_dimension, False),  # dynamic dimension is not fully defined
                (dynamic_dimension_value, True),  # dynamic dimension value is fully defined
                ((dynamic_dimension_value, dynamic_dimension_value), True),  # list with dynamic dimension values is
                # fully defined
                ((dynamic_dimension, 1), False),  # tuple with dynamic dimension is not fully defined
                ([dynamic_dimension, 1], False),  # list with dynamic dimension is not fully defined
                ])
    def test_is_fully_defined(self, data, result):
        assert is_fully_defined(data) == result


class TestShapeArrayTest():
    @pytest.mark.parametrize("data, ref, result",[([1], shape_array([1]), True),
                # if we provide a list with dynamic_dimension_value then it is converted to dynamic dimension
                ([dynamic_dimension_value, 5], gen_masked_array([1, 5], [0]), True),
                # if we provide a list with dynamic_dimension then the generated shape array still have it
                ([7, dynamic_dimension], gen_masked_array([7, 1], [1]), True),
                # negative test to make sure that np.ma.allequal works properly
                ([2], gen_masked_array([1], []), False),
                ])
    def test_shape_array(self, data, ref, result):
        assert strict_compare_tensors(shape_array(data), ref) == result


class TestCompareShapesTest():
    @pytest.mark.parametrize("input1, input2, result",[(gen_masked_array([1, 2, 3], []),
                                                        gen_masked_array([1, 2, 3], []), True),
                (gen_masked_array([4, 2, 3], []), gen_masked_array([1, 2, 3], []), False),
                (gen_masked_array([1, 2], []), gen_masked_array([1, 2, 3], []), False),
                (gen_masked_array([1, 2, 3], []), gen_masked_array([1, 2], []), False),
                (gen_masked_array([1, 2, 3], [1]), gen_masked_array([1, 5, 3], [1]), True),  # [1, d, 3] vs [1, d, 3]
                (gen_masked_array([1, 2, 3], [2]), gen_masked_array([1, 5, 3], [1]), True),  # [1, 2, d] vs [1, d, 3]
                (gen_masked_array([1, 2, 3], []), gen_masked_array([1, 5, 3], [1]), True),  # [1, 2, 3] vs [1, d, 3]
                (gen_masked_array([1, 2, 3], [0]), gen_masked_array([1, 5, 3], []), False),  # [d, 2, 3] vs [1, 5, 3]
                (np.array([1, 2, 3]), gen_masked_array([1, 5, 3], [1]), True),  # [1, 2, 3] vs [1, d, 3]
                (np.array([1, 2]), gen_masked_array([1, 5, 3], [1]), False),
                (np.array([1, 2]), np.array([1, 2]), True),
                (np.array([1, 2]), np.array([3, 2]), False),
                ])
    def test_compare_shapes(self, input1, input2, result):
        assert compatible_shapes(input1, input2) == result


class TestShapeDeleteTest():
    @pytest.mark.parametrize("shape, indices, result",[(gen_masked_array([1, 2, 3], []), [],
                                                        gen_masked_array([1, 2, 3], [])),
                # [1, d, 3] -> [d, 3]. Indices input is a list
                (gen_masked_array([1, 2, 3], [1]), [0], gen_masked_array([2, 3], [0])),
                # [1, d, 3] -> [d, 3]. Indices input is a numpy array
                (gen_masked_array([1, 2, 3], [1]), np.array([0]), gen_masked_array([2, 3], [0])),
                # [1, d, 3] -> [d, 3]. Indices input is a masked array
                (gen_masked_array([1, 2, 3], [1]), gen_masked_array([0], []), gen_masked_array([2, 3], [0])),
                # [1, d, 3] -> [d, 3]. Indices input is a numpy array with scalar
                (gen_masked_array([1, 2, 3], [1]), np.array(0), gen_masked_array([2, 3], [0])),
                # [1, d, 3] -> [d, 3]. Indices input is an integer
                (gen_masked_array([1, 2, 3], [1]), 0, gen_masked_array([2, 3], [0])),  # [1, d, 3] -> [d, 3]
                (gen_masked_array([1, 2, 3, 4], [1]), [0, 2], gen_masked_array([2, 4], [0])),  # [1, d, 3, 4] -> [d, 4]
                (gen_masked_array([1, 2, 3], [1]), [0, 2, 1], gen_masked_array([], [])),  # [1, d, 3] -> []
                (gen_masked_array([1, 2, 3], [1]), [0, 2], gen_masked_array([2], [0])),  # [1, d, 3] -> [d]
                # [1, d, d, 4] -> [d, d]
                (gen_masked_array([1, 2, 3, 4], [1, 2]), [3, 0], gen_masked_array([2, 3], [0, 1])),
                (gen_masked_array([1, 2, 3, 4], [2]), 3, gen_masked_array([1, 2, 3], [2])),  # [1, 2, d, 4] -> [1, 2, d]
                ([1, 2, 3, 4], [1], [1, 3, 4]),  # [1, 2, 3, 4] -> [1, 3, 4]. Input is a regular lists
                (np.array([1, 2, 3, 4]), [1], [1, 3, 4]),  # [1, 2, 3, 4] -> [1, 3, 4]. Input is a regular arrays
                (np.array([1, 2, 3, 4]), [-1, -3], [1, 3]),  # [1, 2, 3, 4] -> [1, 3]. Negative indices
                (np.array([1, 2, 3, 4]), -2, [1, 2, 4]),  # [1, 2, 3, 4] -> [1, 2, 4]. Negative index
                ])
    def test_shape_delete(self, shape, indices, result):
        assert strict_compare_tensors(shape_delete(shape, indices), result)

    def test_shape_delete_raise_exception(self):
        with pytest.raises(Error, match ='.*Incorrect parameter type.*'):
            shape_delete(gen_masked_array([1, 2, 3], []), {})


class TestShapeInsertTest():
    @pytest.mark.parametrize("shape, pos, values, result",[(gen_masked_array([1, 2, 3], []), 1, [5],
                                                            gen_masked_array([1, 5, 2, 3], [])),
                (gen_masked_array([1, 2, 3], [1]), 1, [5], gen_masked_array([1, 5, 2, 3], [2])),
                (gen_masked_array([1, 2, 3], [1]), 1, [dynamic_dimension], gen_masked_array([1, 5, 2, 3], [1, 2])),
                (gen_masked_array([1, 2, 3], [1]), 0, [dynamic_dimension], gen_masked_array([5, 1, 2, 3], [0, 2])),
                (gen_masked_array([1, 2, 3], [1]), np.int64(0), [dynamic_dimension],
                 gen_masked_array([5, 1, 2, 3], [0, 2])),
                (gen_masked_array([1, 2, 3], [1]), 3, [dynamic_dimension], gen_masked_array([1, 2, 3, 5], [1, 3])),
                (gen_masked_array([1, 2, 3], [1]), 3, [dynamic_dimension, dynamic_dimension],
                 gen_masked_array([1, 2, 3, 5, 6], [1, 3, 4])),
                (gen_masked_array([1], [0]), 0, [7, dynamic_dimension], gen_masked_array([7, 5, 2], [1, 2])),
                ])
    def test_shape_insert(self, shape, pos, values, result):
        assert strict_compare_tensors(shape_insert(shape, pos, values), result)

    def test_shape_insert_raise_exception(self):
        with pytest.raises(Error, match='.*Incorrect parameter type.*'):
            shape_insert(gen_masked_array([1, 2, 3], []), 2, {})


class Testmo_array_test():
    @pytest.mark.parametrize("data, result",[(mo_array([2, 3, 5, 7]), np.array([2, 3, 5, 7])),
                (mo_array([2., 3., 5., 7.], dtype=np.float64), np.array([2., 3., 5., 7.])),
                (mo_array([2., 3., 5., 7.]), np.array([2., 3., 5., 7.], dtype=np.float32)),
                ])
    def test_mo_array_positive(self, data, result):
        assert data.dtype == result.dtype

    @pytest.mark.parametrize("data, result",[(mo_array([2., 3., 5., 7.]), np.array([2., 3., 5., 7.])),
                ])
    def test_mo_array_negative(self, data, result):
        assert data.dtype != result.dtype


class clarify_partial_shape_test(unittest.TestCase):

    def test_clarify_1(self):
        actual_result = clarify_partial_shape([shape_array([dynamic_dimension, 10, dynamic_dimension]),
                                              shape_array([4, dynamic_dimension, dynamic_dimension])])
        ref_result = shape_array([4, 10, dynamic_dimension])
        assert strict_compare_tensors(actual_result, ref_result)
